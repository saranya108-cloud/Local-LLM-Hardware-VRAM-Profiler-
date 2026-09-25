/* ------------------------------------------------------------------ *
 * Quantization formats — effective bits per weight including the
 * per-block scale/zero-point metadata that GGUF k-quants carry.
 * `family` lets a backend hide formats without removing them.
 * ------------------------------------------------------------------ */
export const QUANTS = [
  { id: 'fp16', label: 'FP16', bpw: 16, quality: 100, family: 'gguf', note: 'Reference precision. No quality loss, maximum memory.' },
  { id: 'q8_0', label: 'Q8_0', bpw: 8.5, quality: 99, family: 'gguf', note: 'Effectively lossless. Good when VRAM is not the constraint.' },
  { id: 'q6_k', label: 'Q6_K', bpw: 6.56, quality: 98, family: 'gguf', note: 'Near-lossless. The best quality/size point above 4-bit.' },
  { id: 'q4_k_m', label: 'Q4_K_M', bpw: 4.85, quality: 95, family: 'gguf', note: 'The community default. Small, measurable perplexity cost.' },
  { id: 'q3_k_s', label: 'Q3_K_S', bpw: 3.44, quality: 87, family: 'gguf', note: 'Aggressive. Noticeable degradation — a last resort to fit.' },
  { id: 'exl2', label: 'EXL2', bpw: 4.25, quality: 93, family: 'exl2', note: 'ExLlamaV2 @ 4.25 bpw. GPU-only, fastest single-stream decode.' },
];

/* KV cache element precision. Quantized KV is the cheapest way to buy
 * back context length once the weights are already as small as useful. */
export const KV_PRECISIONS = [
  { id: 'fp16', label: 'FP16', bytes: 2, note: 'Default KV precision.' },
  { id: 'q8', label: 'Q8', bytes: 1, note: 'Halves cache size, negligible quality impact.' },
  { id: 'q4', label: 'Q4', bytes: 0.5, note: 'Quarter size. Can degrade long-context recall.' },
];
