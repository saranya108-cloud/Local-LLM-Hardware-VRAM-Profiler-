/* ------------------------------------------------------------------ *
 * Turn selected catalog ids into the numeric bag calculateProfile
 * already understands. The UI does not sum overhead or blend bandwidth.
 * ------------------------------------------------------------------ */
import { ATTENTION_SCRATCH_FACTOR } from './calculations.js';
import { BACKENDS, DEFAULT_BACKEND_ID } from './data/backends.js';
import { DEFAULT_MARGIN_ID, MARGIN_BUCKETS, MARGIN_PROFILES, sumMarginBuckets } from './data/margins.js';
import { DEFAULT_MEMORY_ID, MEMORY_PROFILES } from './data/memory.js';

function requirePositiveFinite(value, label) {
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive finite number`);
  }
  return value;
}

export function resolveAssumptions({
  memoryId = DEFAULT_MEMORY_ID,
  systemBandwidthGBPerSecond,
  marginId = DEFAULT_MARGIN_ID,
  backendId = DEFAULT_BACKEND_ID,
} = {}) {
  const memory = MEMORY_PROFILES.find((profile) => profile.id === memoryId);
  if (!memory) throw new RangeError(`unknown memory profile: ${memoryId}`);
  if (!memory.selectable) {
    throw new RangeError(`memory profile ${memoryId} is not selectable`);
  }

  const systemRamBandwidthGBPerSecond =
    memory.id === 'custom'
      ? requirePositiveFinite(systemBandwidthGBPerSecond, 'custom system bandwidth')
      : requirePositiveFinite(memory.bandwidthGBPerSecond, `memory profile ${memory.id} bandwidth`);

  const margin = MARGIN_PROFILES.find((profile) => profile.id === marginId);
  if (!margin) throw new RangeError(`unknown margin profile: ${marginId}`);
  const baselineOverheadGB = sumMarginBuckets(margin.buckets);

  const backend = BACKENDS.find((item) => item.id === backendId);
  if (!backend) throw new RangeError(`unknown backend: ${backendId}`);

  return {
    systemRamBandwidthGBPerSecond,
    baselineOverheadGB,
    attentionScratchFactor: ATTENTION_SCRATCH_FACTOR,
    cpuOffloadMode: backend.cpuOffload,
    memory: {
      id: memory.id,
      label: memory.label,
      mode: memory.mode,
      note: memory.note,
    },
    backend: {
      id: backend.id,
      label: backend.label,
      note: backend.note,
      cpuOffload: backend.cpuOffload,
    },
    marginBuckets: MARGIN_BUCKETS.map((bucket) => ({
      id: bucket.id,
      label: bucket.label,
      gb: margin.buckets[bucket.id],
      modeled: bucket.modeled,
    })),
  };
}

export function profileInputsFromAssumptions(assumptions) {
  return {
    systemRamBandwidthGBPerSecond: assumptions.systemRamBandwidthGBPerSecond,
    baselineOverheadGB: assumptions.baselineOverheadGB,
    attentionScratchFactor: assumptions.attentionScratchFactor,
    cpuOffloadMode: assumptions.cpuOffloadMode,
  };
}
