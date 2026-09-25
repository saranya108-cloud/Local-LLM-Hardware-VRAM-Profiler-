/* ------------------------------------------------------------------ *
 * Model architectures
 * Layer / head geometry drives the KV cache, so it is tracked per
 * model rather than guessed from the parameter count alone.
 * ------------------------------------------------------------------ */
export const MODELS = [
  {
    id: '3b',
    label: '3B',
    example: 'Llama 3.2 3B',
    params: 3.2,
    layers: 28,
    hidden: 3072,
    heads: 24,
    kvHeads: 8,
    headDim: 128,
  },
  {
    id: '7b',
    label: '7B / 8B',
    example: 'Mistral 7B · Llama 3.1 8B',
    params: 8.0,
    layers: 32,
    hidden: 4096,
    heads: 32,
    kvHeads: 8,
    headDim: 128,
  },
  {
    id: '14b',
    label: '14B',
    example: 'Qwen 2.5 14B',
    params: 14.8,
    layers: 48,
    hidden: 5120,
    heads: 40,
    kvHeads: 8,
    headDim: 128,
  },
  {
    id: '32b',
    label: '32B',
    example: 'Qwen 2.5 32B · QwQ',
    params: 32.8,
    layers: 64,
    hidden: 5120,
    heads: 40,
    kvHeads: 8,
    headDim: 128,
  },
  {
    id: '70b',
    label: '70B',
    example: 'Llama 3.3 70B · DeepSeek R1 Distill',
    params: 70.6,
    layers: 80,
    hidden: 8192,
    heads: 64,
    kvHeads: 8,
    headDim: 128,
  },
  {
    id: 'custom',
    label: 'Custom',
    example: 'Set your own parameter count',
    params: 24,
    layers: null,
    hidden: null,
    heads: null,
    kvHeads: 8,
    headDim: 128,
  },
];

export const CONTEXT_STEPS = [2048, 4096, 8192, 16384, 32768, 65536, 98304, 131072];
