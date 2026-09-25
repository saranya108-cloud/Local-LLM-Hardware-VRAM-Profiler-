# VRAM Calc & Local LLM Profiler

A lightweight, single-file interactive web application for AI engineers, developers, and home lab enthusiasts hosting local open-source models. Quickly estimate VRAM requirements, context overhead, memory bandwidth bottlenecks, and hardware compatibility before downloading model weights.

---

## 🎯 Features

* **Model & Quantization Configuration:** Select target model parameters (7B to 70B+) and precision levels (FP16, Q8_0, Q4_K_M, EXL2, etc.).
* **Dynamic VRAM Allocation Breakdown:** Real-time calculation and visualization of memory distribution across three key zones:
  * **Model Weights:** Static memory required to load parameters.
  * **KV Cache:** Context-dependent memory required for attention key-value states.
  * **CUDA / Runtime Overhead:** Baseline framework memory buffers.
* **Backend Profiles:** Generic, llama.cpp, Ollama, MLX, ExLlamaV2, and vLLM. A backend limits which quantization formats are offered and whether the CPU-offload ceiling is shown; it never changes the weight, KV, or overhead formulas.
* **System Memory Profile:** Choose the bandwidth used for weights that spill out of device memory — Generic DDR (60 GB/s) or a custom GB/s value.
* **Hardware Preset Matching:** Instant pass/fail analysis across popular GPUs and unified memory setups (NVIDIA RTX series, A100/H100, Acer Veriton GN100 / NVIDIA GB10, and Apple Silicon M-series).
* **Theoretical Decode Ceiling:** A bandwidth-derived upper bound on tokens per second, not expected benchmark performance.
* **Actionable Recommendations:** Automated suggestions when VRAM overflows (e.g., flash attention context offloading, lower quantization, layer offloading ratios).

---

## 🚀 Quick Start

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # production bundle in dist/
```

State, the recommendation engine, charts, and UI live in [`src/App.jsx`](src/App.jsx).
The calculation engine is [`src/calculations.js`](src/calculations.js). Hardware,
model, quantization, KV precision, context-step, system-memory, safety-margin, and
backend catalogs live in [`src/data/`](src/data/).
[`src/assumptions.js`](src/assumptions.js) resolves the selected memory, margin,
and backend profiles into plain numbers before they reach the engine. Everything
else is standard Vite + Tailwind scaffolding.

To drop it into an existing project or an online sandbox (StackBlitz,
CodeSandbox), copy `src/App.jsx`, `src/calculations.js`, `src/assumptions.js`, and `src/data/`, and
install its three runtime dependencies:

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
| **Runtime overhead** | Sum of the safety-margin buckets — a `1.5 GB` unallocated CUDA/runtime baseline, with operating system, driver, framework buffers, and temporary peaks at `0 GB` (not modeled yet) — plus 25% of the cache as attention scratch when Flash Attention is disabled |
| **Total required** | `weights + KV cache + runtime overhead` |
| **Theoretical decode ceiling** | `(decimal_bandwidth_GB/s × 10⁹ ÷ 2³⁰) ÷ weight_footprint_GiB` — manufacturer bandwidth is converted from decimal GB/s to GiB/s before division |
| **Offload penalty** | When the backend allows offload and it is viable, weights that spill out of VRAM are read at the selected system-memory bandwidth (Generic DDR at 60 GB/s by default, or a custom GB/s value), blended with device bandwidth as a weighted harmonic mean and then converted to GiB/s |

The theoretical decode ceiling is an upper bound, not a prediction of expected
benchmark performance. CPU offload is viable only when the KV cache plus
runtime memory fit on the device without any model weights; otherwise reducing
context, batch size, or KV precision, or selecting a larger device is required.

Changing the system-memory profile only moves the spill ceiling: a model that
fits entirely on the device keeps the same total and the same device-bandwidth
ceiling. For example, a 70B Q4_K_M model at 16k context on an RTX 4090 needs
54.3 GB either way, with a ceiling of about ≤1.8 tok/s at 60 GB/s and ≤5.0 tok/s
at 180 GB/s. DDR5 desktop and DDR5 server are reserved profile names with no
bandwidth and are not selectable until a sourced figure is added.

### Backend profiles

| Backend | Quantization formats | CPU-offload ceiling |
|---|---|---|
| **Generic** | All catalog formats | Shown when the KV cache plus runtime memory fit |
| **llama.cpp** | GGUF formats in the catalog | Shown when the KV cache plus runtime memory fit |
| **Ollama** | GGUF formats in the catalog | Shown when the KV cache plus runtime memory fit |
| **MLX** | None tagged yet (the estimate keeps generic formulas) | Shown when the KV cache plus runtime memory fit |
| **ExLlamaV2** | EXL2 only | Withheld |
| **vLLM** | GGUF formats in the catalog (vLLM GGUF loading is experimental; EXL2 is not offered) | Withheld — CPU weight offload is not modeled |

No backend adds overhead or scales device bandwidth, and MLX adds no separate
Apple bandwidth figure.

Calculated memory footprints use GiB (2³⁰ bytes), while hardware preset
capacity values are compared as listed. The interface uses the familiar `GB`
label for both. Apple Silicon unified-memory presets list the
practically allocatable share, not total system memory. The Acer Veriton GN100
/ NVIDIA GB10 preset models 128 GB of published physical LPDDR5X unified
memory, not a verified LLM-allocatable amount; OS, display, driver, and
runtime share that pool. Apple Silicon and GB10 overflow uses the selected
system-memory profile (Generic DDR at 60 GB/s unless another is chosen) — that
is the existing discrete-GPU spill estimator, not a second memory pool on these
machines. Every assumption is restated in the app's
own **Model assumptions** panel and in the GB10 preset note.
