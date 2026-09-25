/* ------------------------------------------------------------------ *
 * Runtime overhead buckets. The unallocated baseline is today's
 * CUDA/runtime lump. Zero means the bucket is not modeled yet.
 * Attention scratch and the 1.2× weight factor stay outside this list.
 * ------------------------------------------------------------------ */
export const MARGIN_BUCKETS = [
  {
    id: 'baseline',
    label: 'Unallocated baseline',
    gb: 1.5,
    modeled: true,
  },
  {
    id: 'os',
    label: 'Operating system',
    gb: 0,
    modeled: false,
  },
  {
    id: 'driver',
    label: 'Driver',
    gb: 0,
    modeled: false,
  },
  {
    id: 'framework',
    label: 'Framework buffers',
    gb: 0,
    modeled: false,
  },
  {
    id: 'peaks',
    label: 'Temporary peaks',
    gb: 0,
    modeled: false,
  },
];

export const DEFAULT_MARGIN_ID = 'default';

export const MARGIN_PROFILES = [
  {
    id: DEFAULT_MARGIN_ID,
    label: 'Default',
    buckets: Object.fromEntries(MARGIN_BUCKETS.map((bucket) => [bucket.id, bucket.gb])),
  },
];

export function sumMarginBuckets(buckets) {
  return MARGIN_BUCKETS.reduce((total, bucket) => {
    const gb = buckets[bucket.id];
    if (!Number.isFinite(gb) || gb < 0) {
      throw new RangeError(`margin bucket ${bucket.id} must be a non-negative finite number`);
    }
    return total + gb;
  }, 0);
}
