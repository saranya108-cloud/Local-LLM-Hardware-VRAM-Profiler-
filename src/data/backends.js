/* ------------------------------------------------------------------ *
 * Backend profiles filter runtime choices. They do not replace the
 * weight or KV formulas, add overhead, or scale device bandwidth.
 * ------------------------------------------------------------------ */
import { CPU_OFFLOAD_DISABLED, CPU_OFFLOAD_WEIGHTS_IF_KV_FITS } from '../calculations.js';
import { QUANTS } from './quantization.js';

export const DEFAULT_BACKEND_ID = 'generic';

export const BACKENDS = [
  {
    id: 'generic',
    label: 'Generic',
    quantFamily: 'all',
    cpuOffload: CPU_OFFLOAD_WEIGHTS_IF_KV_FITS,
    note: 'Current formulas and the full quantization list. CPU offload is allowed when the KV cache plus runtime memory fit.',
  },
  {
    id: 'llama.cpp',
    label: 'llama.cpp',
    quantFamily: 'gguf',
    cpuOffload: CPU_OFFLOAD_WEIGHTS_IF_KV_FITS,
    note: 'GGUF formats already in the catalog. Same fit and bandwidth math as generic.',
  },
  {
    id: 'ollama',
    label: 'Ollama',
    quantFamily: 'gguf',
    cpuOffload: CPU_OFFLOAD_WEIGHTS_IF_KV_FITS,
    note: 'Same quantization set and math as llama.cpp, with no separate overhead.',
  },
  {
    id: 'mlx',
    label: 'MLX',
    quantFamily: 'mlx',
    cpuOffload: CPU_OFFLOAD_WEIGHTS_IF_KV_FITS,
    note: 'No formats are tagged for MLX yet. This profile adds no Apple bandwidth and no second memory pool.',
  },
  {
    id: 'exllamav2',
    label: 'ExLlamaV2',
    quantFamily: 'exl2',
    cpuOffload: CPU_OFFLOAD_DISABLED,
    note: 'EXL2 only. CPU offload is off, so the spill ceiling is withheld. Overhead is unchanged.',
  },
  {
    id: 'vllm',
    label: 'vLLM',
    quantFamily: 'gguf',
    cpuOffload: CPU_OFFLOAD_DISABLED,
    note: 'GGUF formats already in the catalog (vLLM GGUF loading is experimental). EXL2 is not offered. CPU weight offload is not modeled, so the spill ceiling is withheld. Tensor parallel keeps the preset device bandwidth.',
  },
];

export function quantsForBackend(backend) {
  if (backend.quantFamily === 'all') return QUANTS;
  return QUANTS.filter((quant) => quant.family === backend.quantFamily);
}

export function preferredQuantId(backend, currentQuantId, fallbackId = 'q4_k_m') {
  const allowed = quantsForBackend(backend);
  if (allowed.some((quant) => quant.id === currentQuantId)) return currentQuantId;
  const fallback = allowed.find((quant) => quant.id === fallbackId);
  if (fallback) return fallback.id;
  return allowed[0]?.id ?? null;
}
