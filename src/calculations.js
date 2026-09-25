export const GIB = 1024 ** 3;
export const SYSTEM_RAM_BANDWIDTH_GB_PER_SECOND = 60;
export const BASELINE_OVERHEAD_GB = 1.5;
export const ATTENTION_SCRATCH_FACTOR = 0.25;
export const CPU_OFFLOAD_WEIGHTS_IF_KV_FITS = 'weights-if-kv-fits';
export const CPU_OFFLOAD_DISABLED = 'disabled';

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
  baselineOverheadGB = BASELINE_OVERHEAD_GB,
  attentionScratchFactor = ATTENTION_SCRATCH_FACTOR,
  cpuOffloadMode = CPU_OFFLOAD_WEIGHTS_IF_KV_FITS,
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
  if (!Number.isFinite(baselineOverheadGB) || baselineOverheadGB < 0) {
    throw new RangeError('baselineOverheadGB must be a non-negative finite number');
  }
  if (!Number.isFinite(attentionScratchFactor) || attentionScratchFactor < 0) {
    throw new RangeError('attentionScratchFactor must be a non-negative finite number');
  }
  if (
    cpuOffloadMode !== CPU_OFFLOAD_WEIGHTS_IF_KV_FITS &&
    cpuOffloadMode !== CPU_OFFLOAD_DISABLED
  ) {
    throw new TypeError('cpuOffloadMode must be "weights-if-kv-fits" or "disabled"');
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

  // Without flash attention the backend materialises attention scores.
  // The default scratch factor is a quarter of the cache.
  const attentionScratch = flashAttention ? 0 : kv * attentionScratchFactor;
  const overhead = baselineOverheadGB + attentionScratch;

  const total = weights + kv + overhead;
  const remaining = capacityGB - total;
  const utilisation = (total / capacityGB) * 100;
  const fits = total <= capacityGB;

  // Hardware vendors report decimal GB/s, while weight footprints use GiB.
  // Decode is memory-bandwidth bound: each token streams the weights once.
  const bandwidthGiBPerSecond = decimalGBToGiB(bandwidthGBPerSecond);
  const idealTps = bandwidthGiBPerSecond / weights;

  // CPU offload can only help when the backend allows it and the
  // device-resident KV cache and runtime memory fit before any model
  // weights are loaded.
  const nonWeightMemory = kv + overhead;
  const cpuOffloadPossible =
    cpuOffloadMode === CPU_OFFLOAD_WEIGHTS_IF_KV_FITS && nonWeightMemory <= capacityGB;

  // Overflow spills layers to system RAM, so the effective read
  // bandwidth becomes a weighted harmonic mean of device memory and
  // the selected system bandwidth. A disabled backend withholds that
  // ceiling; a fully resident model still uses device bandwidth.
  const overflow = Math.max(0, total - capacityGB);
  const offloadFraction = weights > 0 ? Math.min(1, Math.max(0, overflow / weights)) : 0;
  let tps = null;
  if (offloadFraction === 0 || cpuOffloadPossible) {
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
  const kvSlope = kvPerTokenGB * batchSize * (flashAttention ? 1 : 1 + attentionScratchFactor);
  const maxContext =
    kvSlope > 0 ? Math.max(0, (capacityGB - weights - baselineOverheadGB) / kvSlope) : 0;

  return {
    weightsRaw,
    weights,
    kv,
    kvPerTokenGB,
    attentionScratch,
    baselineOverheadGB,
    attentionScratchFactor,
    cpuOffloadMode,
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
