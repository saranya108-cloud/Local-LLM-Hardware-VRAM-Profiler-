# VRAM Calc & Local LLM Profiler

A lightweight, single-file interactive web application for AI engineers, developers, and home lab enthusiasts hosting local open-source models. Quickly estimate VRAM requirements, context overhead, memory bandwidth bottlenecks, and hardware compatibility before downloading model weights.

---

## 🎯 Features

* **Model & Quantization Configuration:** Select target model parameters (7B to 70B+) and precision levels (FP16, Q8_0, Q4_K_M, EXL2, etc.).
* **Dynamic VRAM Allocation Breakdown:** Real-time calculation and visualization of memory distribution across three key zones:
  * **Model Weights:** Static memory required to load parameters.
  * **KV Cache:** Context-dependent memory required for attention key-value states.
  * **CUDA / Runtime Overhead:** Baseline framework memory buffers.
* **Hardware Preset Matching:** Instant pass/fail analysis across popular GPUs and unified memory setups (NVIDIA RTX series, A100/H100, and Apple Silicon M-series).
* **Throughput Estimation:** Approximate tokens-per-second output based on memory bandwidth constraints (GB/s).
* **Actionable Recommendations:** Automated suggestions when VRAM overflows (e.g., flash attention context offloading, lower quantization, layer offloading ratios).

---

## 🚀 Quick Start

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # production bundle in dist/
```

The entire application — state, calculations, recommendation engine, charts, and
UI — lives in a single component file, [`src/App.jsx`](src/App.jsx). Everything
else is standard Vite + Tailwind scaffolding.

To drop it into an existing project or an online sandbox (StackBlitz,
CodeSandbox), copy `src/App.jsx` and install its three runtime dependencies:

```bash
npm install react react-dom lucide-react recharts
```

Tailwind CSS v4 supplies the utility classes; the range-input and scrollbar
styling lives in `src/index.css`.

---

## 🧮 How the numbers are derived

| Quantity | Formula |
|---|---|
| **Model weights** | `params × (bits_per_weight ÷ 8) × 1.2` — the 1.2 factor covers allocator padding and fragmentation on load |
| **KV cache** | `2 × layers × kv_heads × head_dim × context × batch × bytes_per_element` — grouped-query attention is assumed (8 KV heads) |
| **Runtime overhead** | `1.5 GB` CUDA/runtime baseline, plus 25% of the cache as attention scratch when Flash Attention is disabled |
| **Total required** | `weights + KV cache + runtime overhead` |
| **Tokens/sec** | `memory_bandwidth ÷ weight_footprint` — decode is bandwidth-bound, so this is a ceiling rather than a benchmark |
| **Offload penalty** | Weights that spill out of VRAM are read at an assumed 60 GB/s of system memory bandwidth, blended with GPU bandwidth as a weighted harmonic mean |

`GB` means GiB (2³⁰ bytes) throughout. Unified-memory presets list the
practically allocatable share, not total system memory. Every assumption is
restated in the app's own **Model assumptions** panel.
