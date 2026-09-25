/* ------------------------------------------------------------------ *
 * Hardware presets
 * `vram` is the capacity the estimator compares against. Discrete GPU
 * presets use advertised VRAM. Apple Silicon unified-memory presets list
 * the practical allocation limit, not total system memory. The GB10
 * preset uses published physical unified memory, not a verified
 * allocatable amount.
 * ------------------------------------------------------------------ */
export const DEFAULT_HARDWARE_ID = 'rtx4090';

export const HARDWARE = [
  { id: 'rtx3060', label: 'RTX 3060 12GB', vram: 12, bandwidth: 360, kind: 'gpu', note: 'Entry-level CUDA card.' },
  { id: 'rtx4060ti', label: 'RTX 4060 Ti 16GB', vram: 16, bandwidth: 288, kind: 'gpu', note: 'Roomy VRAM, narrow memory bus.' },
  { id: 'rtx3090', label: 'RTX 3090 24GB', vram: 24, bandwidth: 936, kind: 'gpu', note: 'The used-market local-LLM workhorse.' },
  { id: 'rtx4090', label: 'RTX 4090 24GB', vram: 24, bandwidth: 1008, kind: 'gpu', note: 'Fastest consumer 24GB card.' },
  { id: 'rtx5090', label: 'RTX 5090 32GB', vram: 32, bandwidth: 1792, kind: 'gpu', note: 'GDDR7. Huge bandwidth uplift.' },
  { id: 'dual3090', label: 'Dual RTX 3090 48GB', vram: 48, bandwidth: 936, kind: 'multi', note: 'Tensor-split across 2 cards.' },
  { id: 'dual4090', label: 'Dual RTX 4090 48GB', vram: 48, bandwidth: 1008, kind: 'multi', note: 'Tensor-split across 2 cards.' },
  { id: 'a100_40', label: 'A100 40GB', vram: 40, bandwidth: 1555, kind: 'dc', note: 'HBM2e datacenter accelerator.' },
  { id: 'a100_80', label: 'A100 80GB', vram: 80, bandwidth: 2039, kind: 'dc', note: 'HBM2e, 80GB configuration.' },
  { id: 'h100_80', label: 'H100 80GB', vram: 80, bandwidth: 3350, kind: 'dc', note: 'HBM3. Highest bandwidth listed.' },
  { id: 'gb10', label: 'Acer Veriton GN100 (GB10 128GB unified)', vram: 128, bandwidth: 273, kind: 'unified', note: 'NVIDIA GB10 Grace Blackwell. 128 GB LPDDR5X coherent unified memory shared by CPU and GPU — published physical capacity, not dedicated VRAM and not a verified LLM-allocatable amount. OS, display, driver, and runtime share this pool. 273 GB/s is NVIDIA theoretical peak. Distinct from Apple Silicon. Overflow may still show the generic CPU-offload / 60 GB/s DDR estimate; that is the existing discrete-GPU estimator, not a second memory pool on this machine.' },
  { id: 'm4pro64', label: 'Mac Studio 64GB (M4 Pro)', vram: 48, bandwidth: 273, kind: 'unified', note: '64GB unified · ~48GB allocatable.' },
  { id: 'm4max64', label: 'Mac Studio 64GB (M4 Max)', vram: 48, bandwidth: 546, kind: 'unified', note: '64GB unified · ~48GB allocatable.' },
  { id: 'm3ultra128', label: 'Mac Studio 128GB (M3 Ultra)', vram: 96, bandwidth: 819, kind: 'unified', note: '128GB unified · ~96GB allocatable.' },
  { id: 'custom', label: 'Custom hardware', vram: 24, bandwidth: 900, kind: 'gpu', note: 'Dial in your own VRAM and bandwidth.' },
];
