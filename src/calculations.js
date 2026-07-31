export const GIB = 1024 ** 3;
export const SYSTEM_RAM_BANDWIDTH_GB_PER_SECOND = 60;
export const BASELINE_OVERHEAD_GB = 1.5;

export function decimalGBToGiB(value) {
  if (!Number.isFinite(value)) throw new TypeError('value must be a finite number');
  return value * 1e9 / GIB;
}

export function calculateProfile({
  arch,
  bitsPerWeight,
  kvBytesPerElement,
  context,
  batchSize,
  flashAttention,
  capacityGB,
  bandwidthGBPerSecond,
  systemRamBandwidthGBPerSecond = SYSTEM_RAM_BANDWIDTH_GB_PER_SECOND,
}) {
  if (!arch || typeof arch !== 'object') throw new TypeError('arch must be an object');

  const positiveInputs = {
    'arch.params': arch.params,
    'arch.layers': arch.layers,
    'arch.kvHeads': arch.kvHeads,
    'arch.headDim': arch.headDim,
    bitsPerWeight,
    kvBytesPerElement,
    batchSize,
    capacityGB,
    bandwidthGBPerSecond,
    systemRamBandwidthGBPerSecond,
  };
  for (const [name, value] of Object.entries(positiveInputs)) {
    if (!Number.isFinite(value) || value <= 0) {
      throw new RangeError(`${name} must be a positive finite number`);
    }
  }
  if (!Number.isFinite(context) || context < 0) {
    throw new RangeError('context must be a non-negative finite number');
  }
  if (typeof flashAttention !== 'boolean') {
    throw new TypeError('flashAttention must be a boolean');
  }

  // Model weights: params x bytes-per-weight, plus a 1.2x allocator /
  // fragmentation factor for the loaded tensors.
  const weightsRaw = (arch.params * 1e9 * (bitsPerWeight / 8)) / GIB;
  const weights = weightsRaw * 1.2;

  // KV cache: 2 tensors (K and V) x layers x kv-heads x head-dim per token.
  // Grouped-query attention means kv-heads, not attention heads.
  const kvBytesPerToken = 2 * arch.layers * arch.kvHeads * arch.headDim * kvBytesPerElement;
  const kvPerTokenGB = kvBytesPerToken / GIB;
  const kv = kvPerTokenGB * context * batchSize;

  // Without flash attention the backend materialises attention scores,
  // costing roughly a quarter of the cache again in scratch buffers.
  const attentionScratch = flashAttention ? 0 : kv * 0.25;
  const overhead = BASELINE_OVERHEAD_GB + attentionScratch;

  const total = weights + kv + overhead;
  const remaining = capacityGB - total;
  const utilisation = (total / capacityGB) * 100;
  const fits = total <= capacityGB;

  // Hardware vendors report decimal GB/s, while weight footprints use GiB.
  // Decode is memory-bandwidth bound: each token streams the weights once.
  const bandwidthGiBPerSecond = decimalGBToGiB(bandwidthGBPerSecond);
  const idealTps = bandwidthGiBPerSecond / weights;

  // CPU offload can only help when the device-resident KV cache and runtime
  // memory fit before any model weights are loaded.
  const nonWeightMemory = kv + overhead;
  const cpuOffloadPossible = nonWeightMemory <= capacityGB;

  // Overflow spills layers to system RAM, so the effective read
  // bandwidth becomes a weighted harmonic mean of VRAM and DDR.
  const overflow = Math.max(0, total - capacityGB);
  const offloadFraction = weights > 0 ? Math.min(1, Math.max(0, overflow / weights)) : 0;
  let tps = null;
  if (cpuOffloadPossible) {
    const effectiveBandwidthGBPerSecond =
      offloadFraction > 0
        ? 1 / ((1 - offloadFraction) / bandwidthGBPerSecond + offloadFraction / systemRamBandwidthGBPerSecond)
        : bandwidthGBPerSecond;
    tps = decimalGBToGiB(effectiveBandwidthGBPerSecond) / weights;
  }

  // Batched decode amortises the weight read across streams, but
  // attention and scheduling keep it sub-linear.
  const aggregateTps = tps === null ? null : tps * Math.pow(batchSize, 0.85);

  // Largest context that still fits, at the current everything-else.
  const kvSlope = kvPerTokenGB * batchSize * (flashAttention ? 1 : 1.25);
  const maxContext =
    kvSlope > 0 ? Math.max(0, (capacityGB - weights - BASELINE_OVERHEAD_GB) / kvSlope) : 0;

  return {
    weightsRaw,
    weights,
    kv,
    kvPerTokenGB,
    attentionScratch,
    overhead,
    total,
    remaining,
    utilisation,
    fits,
    bandwidthGiBPerSecond,
    idealTps,
    tps,
    aggregateTps,
    offloadFraction,
    nonWeightMemory,
    cpuOffloadPossible,
    maxContext,
  };
}
