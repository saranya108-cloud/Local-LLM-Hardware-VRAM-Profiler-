import React, { useMemo, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Boxes,
  CheckCircle2,
  ChevronDown,
  Cpu,
  Database,
  Gauge,
  HardDrive,
  Info,
  Layers,
  Lightbulb,
  MemoryStick,
  Rows3,
  Server,
  SlidersHorizontal,
  Terminal,
  TrendingUp,
  XCircle,
  Zap,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  BASELINE_OVERHEAD_GB,
  GIB,
  SYSTEM_RAM_BANDWIDTH_GB_PER_SECOND,
  calculateProfile,
} from './calculations.js';

/* ------------------------------------------------------------------ *
 * Design tokens — dark terminal theme.
 * Series slots 1-3 are the validated categorical order for a dark
 * surface (CVD-safe, all >= 3:1 against #14141a).
 * ------------------------------------------------------------------ */
const T = {
  page: '#0a0a0c',
  surface: '#14141a',
  surfaceRaised: '#1c1c24',
  surfaceInput: '#101017',
  border: 'rgba(255,255,255,0.08)',
  borderStrong: 'rgba(255,255,255,0.14)',
  ink: '#f2f2f5',
  inkSecondary: '#b4b4c2',
  inkMuted: '#8a8a99',
  grid: '#23232c',
  axis: '#33333f',
  weights: '#3987e5',
  kv: '#d95926',
  overhead: '#199e70',
  good: '#0ca30c',
  warning: '#fab219',
  critical: '#d03b3b',
};

/* ------------------------------------------------------------------ *
 * Model architectures
 * Layer / head geometry drives the KV cache, so it is tracked per
 * model rather than guessed from the parameter count alone.
 * ------------------------------------------------------------------ */
const MODELS = [
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

/* ------------------------------------------------------------------ *
 * Quantization formats — effective bits per weight including the
 * per-block scale/zero-point metadata that GGUF k-quants carry.
 * ------------------------------------------------------------------ */
const QUANTS = [
  { id: 'fp16', label: 'FP16', bpw: 16, quality: 100, note: 'Reference precision. No quality loss, maximum memory.' },
  { id: 'q8_0', label: 'Q8_0', bpw: 8.5, quality: 99, note: 'Effectively lossless. Good when VRAM is not the constraint.' },
  { id: 'q6_k', label: 'Q6_K', bpw: 6.56, quality: 98, note: 'Near-lossless. The best quality/size point above 4-bit.' },
  { id: 'q4_k_m', label: 'Q4_K_M', bpw: 4.85, quality: 95, note: 'The community default. Small, measurable perplexity cost.' },
  { id: 'q3_k_s', label: 'Q3_K_S', bpw: 3.44, quality: 87, note: 'Aggressive. Noticeable degradation — a last resort to fit.' },
  { id: 'exl2', label: 'EXL2', bpw: 4.25, quality: 93, note: 'ExLlamaV2 @ 4.25 bpw. GPU-only, fastest single-stream decode.' },
];

/* KV cache element precision. Quantized KV is the cheapest way to buy
 * back context length once the weights are already as small as useful. */
const KV_PRECISIONS = [
  { id: 'fp16', label: 'FP16', bytes: 2, note: 'Default KV precision.' },
  { id: 'q8', label: 'Q8', bytes: 1, note: 'Halves cache size, negligible quality impact.' },
  { id: 'q4', label: 'Q4', bytes: 0.5, note: 'Quarter size. Can degrade long-context recall.' },
];

/* ------------------------------------------------------------------ *
 * Hardware presets
 * `vram` is the capacity the estimator compares against. Discrete GPU
 * presets use advertised VRAM. Apple Silicon unified-memory presets list
 * the practical allocation limit, not total system memory. The GB10
 * preset uses published physical unified memory, not a verified
 * allocatable amount.
 * ------------------------------------------------------------------ */
const HARDWARE = [
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

const CONTEXT_STEPS = [2048, 4096, 8192, 16384, 32768, 65536, 98304, 131072];

/* ------------------------------------------------------------------ *
 * Formatting helpers
 * ------------------------------------------------------------------ */
const fmtGB = (v) => `${v < 10 ? v.toFixed(2) : v.toFixed(1)} GB`;
const fmtCtx = (n) => (n >= 1024 ? `${Math.round(n / 1024)}k` : `${n}`);
const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
const fmtTps = (n) => (n >= 100 ? Math.round(n).toString() : n.toFixed(1));

/* Round axis ticks to clean numbers so the scale reads at a glance
 * instead of landing on whatever the data max happens to be. */
function niceTicks(max, target = 5) {
  if (!(max > 0)) return [0];
  const raw = max / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const out = [];
  for (let v = 0; v <= max + 1e-9; v += step) out.push(Number(v.toFixed(4)));
  return out;
}

/* Derive plausible layer/hidden geometry for a custom parameter count.
 * Transformer params ~= 12 * layers * hidden^2, and production models
 * hold hidden/layers near 110, which pins both dimensions. */
function deriveGeometry(params) {
  const layers = Math.max(4, Math.round(Math.cbrt((params * 1e9) / 145200)));
  const hidden = Math.max(512, Math.round((110 * layers) / 128) * 128);
  return { layers, hidden, heads: Math.max(8, Math.round(hidden / 128)), kvHeads: 8, headDim: 128 };
}

/* ------------------------------------------------------------------ *
 * UI primitives
 * ------------------------------------------------------------------ */
function Panel({ title, icon: Icon, action, children, className = '' }) {
  return (
    <section
      className={`rounded-xl border bg-[#14141a] ${className}`}
      style={{ borderColor: T.border }}
    >
      {title && (
        <header
          className="flex items-center justify-between gap-3 border-b px-4 py-3"
          style={{ borderColor: T.border }}
        >
          <h2 className="flex items-center gap-2 font-mono text-[11px] font-semibold uppercase tracking-[0.14em] text-[#b4b4c2]">
            {Icon && <Icon size={14} strokeWidth={2} className="text-[#8a8a99]" aria-hidden="true" />}
            {title}
          </h2>
          {action}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

function Field({ label, hint, children, htmlFor }) {
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between gap-3">
        <label
          htmlFor={htmlFor}
          className="font-mono text-[11px] uppercase tracking-[0.12em] text-[#8a8a99]"
        >
          {label}
        </label>
        {hint && <span className="font-mono text-[11px] text-[#b4b4c2] tabular-nums">{hint}</span>}
      </div>
      {children}
    </div>
  );
}

function Select({ id, value, onChange, options }) {
  return (
    <div className="relative">
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full appearance-none rounded-lg border bg-[#101017] px-3 py-2.5 pr-9 text-sm text-[#f2f2f5] outline-none transition focus:border-[#3987e5] focus:ring-1 focus:ring-[#3987e5]"
        style={{ borderColor: T.border }}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value} className="bg-[#14141a]">
            {o.label}
          </option>
        ))}
      </select>
      <ChevronDown
        size={15}
        aria-hidden="true"
        className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-[#8a8a99]"
      />
    </div>
  );
}

function Segmented({ options, value, onChange, columns = 3, ariaLabel }) {
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className="grid gap-1.5"
      style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
    >
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(o.value)}
            className={`rounded-lg border px-2 py-2 font-mono text-xs transition focus:outline-none focus-visible:ring-1 focus-visible:ring-[#3987e5] ${
              active
                ? 'border-[#3987e5] bg-[#3987e5]/15 text-[#f2f2f5]'
                : 'bg-[#101017] text-[#b4b4c2] hover:border-[rgba(255,255,255,0.2)] hover:text-[#f2f2f5]'
            }`}
            style={active ? undefined : { borderColor: T.border }}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

function Slider({ id, min, max, step = 1, value, onChange }) {
  return (
    <input
      id={id}
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="mt-1"
    />
  );
}

function NumberInput({ id, value, onChange, min, max, step = 1, suffix }) {
  return (
    <div
      className="flex items-center rounded-lg border bg-[#101017] focus-within:border-[#3987e5] focus-within:ring-1 focus-within:ring-[#3987e5]"
      style={{ borderColor: T.border }}
    >
      <input
        id={id}
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const n = Number(e.target.value);
          if (Number.isFinite(n)) onChange(Math.min(max, Math.max(min, n)));
        }}
        className="w-full bg-transparent px-3 py-2.5 font-mono text-sm text-[#f2f2f5] outline-none tabular-nums"
      />
      {suffix && <span className="pr-3 font-mono text-xs text-[#8a8a99]">{suffix}</span>}
    </div>
  );
}

function Toggle({ id, checked, onChange, label, description }) {
  return (
    <button
      id={id}
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="flex w-full items-start gap-3 rounded-lg border bg-[#101017] p-3 text-left transition hover:border-[rgba(255,255,255,0.2)] focus:outline-none focus-visible:ring-1 focus-visible:ring-[#3987e5]"
      style={{ borderColor: checked ? T.weights : T.border }}
    >
      <span
        className="mt-0.5 flex h-4 w-7 shrink-0 items-center rounded-full p-0.5 transition"
        style={{ background: checked ? T.weights : '#2a2a34' }}
      >
        <span
          className="h-3 w-3 rounded-full bg-white transition"
          style={{ transform: checked ? 'translateX(12px)' : 'translateX(0)' }}
        />
      </span>
      <span className="min-w-0">
        <span className="block font-mono text-xs text-[#f2f2f5]">{label}</span>
        <span className="mt-0.5 block text-[11px] leading-snug text-[#8a8a99]">{description}</span>
      </span>
    </button>
  );
}

function MetricCard({ icon: Icon, label, value, unit, sub, tone = 'neutral', hero = false }) {
  const toneColor =
    tone === 'good' ? T.good : tone === 'warning' ? T.warning : tone === 'critical' ? T.critical : T.ink;
  return (
    <div className="rounded-xl border bg-[#14141a] p-4" style={{ borderColor: T.border }}>
      <div className="flex items-center gap-2">
        <Icon size={14} strokeWidth={2} aria-hidden="true" style={{ color: T.inkMuted }} />
        <span className="font-mono text-[11px] uppercase tracking-[0.12em] text-[#8a8a99]">{label}</span>
      </div>
      <div className="mt-2 flex items-baseline gap-1.5">
        <span
          className={`font-semibold leading-none ${hero ? 'text-[44px] sm:text-5xl' : 'text-3xl'}`}
          style={{ color: toneColor }}
        >
          {value}
        </span>
        {unit && <span className="text-sm font-medium text-[#8a8a99]">{unit}</span>}
      </div>
      {sub && <p className="mt-2 text-[11px] leading-snug text-[#b4b4c2]">{sub}</p>}
    </div>
  );
}

function LegendSwatch({ color, label, value }) {
  return (
    <div className="flex items-center gap-2">
      <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: color }} aria-hidden="true" />
      <span className="font-mono text-[11px] text-[#b4b4c2]">{label}</span>
      {value && <span className="font-mono text-[11px] text-[#8a8a99] tabular-nums">{value}</span>}
    </div>
  );
}

function ChartTooltip({ active, payload, labelText }) {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded-lg border bg-[#1c1c24] px-3 py-2 shadow-xl"
      style={{ borderColor: T.borderStrong }}
    >
      {labelText && (
        <p className="mb-1 font-mono text-[10px] uppercase tracking-[0.1em] text-[#8a8a99]">{labelText}</p>
      )}
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center gap-2 py-0.5">
          <span className="h-2 w-2 rounded-sm" style={{ background: p.color || p.stroke }} aria-hidden="true" />
          <span className="font-mono text-[11px] text-[#b4b4c2]">{p.name}</span>
          <span className="ml-auto pl-3 font-mono text-[11px] text-[#f2f2f5] tabular-nums">
            {fmtGB(p.value)}
          </span>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * App
 * ------------------------------------------------------------------ */
export default function App() {
  const [modelId, setModelId] = useState('7b');
  const [customParams, setCustomParams] = useState(24);
  const [quantId, setQuantId] = useState('q4_k_m');
  const [kvPrecisionId, setKvPrecisionId] = useState('fp16');
  const [ctxIndex, setCtxIndex] = useState(3); // 16k
  const [batchSize, setBatchSize] = useState(1);
  const [hardwareId, setHardwareId] = useState('rtx4090');
  const [customVram, setCustomVram] = useState(24);
  const [customBandwidth, setCustomBandwidth] = useState(900);
  const [flashAttention, setFlashAttention] = useState(true);

  /* ---------------- resolved configuration ---------------- */
  const model = MODELS.find((m) => m.id === modelId);
  const quant = QUANTS.find((q) => q.id === quantId);
  const kvPrecision = KV_PRECISIONS.find((k) => k.id === kvPrecisionId);
  const hardwarePreset = HARDWARE.find((h) => h.id === hardwareId);
  const context = CONTEXT_STEPS[ctxIndex];

  const arch = useMemo(() => {
    if (modelId === 'custom') {
      const geo = deriveGeometry(customParams);
      return { params: customParams, ...geo, derived: true };
    }
    return {
      params: model.params,
      layers: model.layers,
      hidden: model.hidden,
      heads: model.heads,
      kvHeads: model.kvHeads,
      headDim: model.headDim,
      derived: false,
    };
  }, [modelId, customParams, model]);

  const capacityGB = hardwareId === 'custom' ? customVram : hardwarePreset.vram;
  const bandwidthGBPerSecond = hardwareId === 'custom' ? customBandwidth : hardwarePreset.bandwidth;
  const capacityLabel =
    hardwareId === 'gb10'
      ? 'Modeled unified memory'
      : hardwarePreset.kind === 'unified'
        ? 'Usable unified memory'
        : 'Usable VRAM';

  /* ---------------- core calculation ---------------- */
  const calc = useMemo(
    () =>
      calculateProfile({
        arch,
        bitsPerWeight: quant.bpw,
        kvBytesPerElement: kvPrecision.bytes,
        context,
        batchSize,
        flashAttention,
        capacityGB,
        bandwidthGBPerSecond,
      }),
    [arch, quant, kvPrecision, context, batchSize, flashAttention, capacityGB, bandwidthGBPerSecond]
  );

  /* ---------------- chart data ---------------- */
  const stackData = [
    {
      name: 'VRAM',
      weights: Number(calc.weights.toFixed(3)),
      kv: Number(calc.kv.toFixed(3)),
      overhead: Number(calc.overhead.toFixed(3)),
    },
  ];
  const axisMax = Math.max(calc.total, capacityGB) * 1.12;

  const contextCurve = useMemo(() => {
    const slope = calc.kvPerTokenGB * batchSize * (flashAttention ? 1 : 1.25);
    return CONTEXT_STEPS.map((c) => ({
      context: c,
      label: fmtCtx(c),
      total: Number((calc.weights + BASELINE_OVERHEAD_GB + slope * c).toFixed(3)),
    }));
  }, [calc.kvPerTokenGB, calc.weights, batchSize, flashAttention]);

  // Keep the capacity reference line inside the plot even when the curve
  // never reaches it, so the pass/fail margin stays visible.
  const curveMax = Math.max(...contextCurve.map((d) => d.total), capacityGB) * 1.08;

  /* ---------------- recommendations ---------------- */
  const recommendations = useMemo(() => {
    const out = [];
    const weightsFor = (q) => ((arch.params * 1e9 * (q.bpw / 8)) / GIB) * 1.2;

    if (!calc.fits) {
      const deficit = calc.total - capacityGB;

      // Cheaper quantization formats, ranked by how much they recover.
      // Prefer the highest-quality format that closes the gap on its own;
      // if none does, still surface the biggest saving available and say
      // plainly how much is left over.
      const cheaper = QUANTS.filter((q) => q.bpw < quant.bpw).sort((a, b) => b.bpw - a.bpw);
      const closing = cheaper.find((q) => calc.weights - weightsFor(q) >= deficit);
      const pick = closing || cheaper[cheaper.length - 1];
      if (pick) {
        const saved = calc.weights - weightsFor(pick);
        const remainder = deficit - saved;
        out.push({
          tone: 'critical',
          icon: Boxes,
          title: `Switch ${quant.label} → ${pick.label}`,
          detail: closing
            ? `Recovers ${fmtGB(saved)} (${Math.round((saved / calc.weights) * 100)}% of weight memory) and brings the total to ${fmtGB(calc.total - saved)} — inside ${fmtGB(capacityGB)}. Quality index ${quant.quality} → ${pick.quality}.`
            : `The largest single saving available: ${fmtGB(saved)} (${Math.round((saved / calc.weights) * 100)}% of weight memory), quality index ${quant.quality} → ${pick.quality}. Still ${fmtGB(remainder)} over — combine it with the steps below.`,
          action: () => setQuantId(pick.id),
          actionLabel: `Apply ${pick.label}`,
        });
      }

      // Quantized KV cache.
      if (kvPrecisionId !== 'q4' && calc.kv > 0.5) {
        const target = kvPrecisionId === 'fp16' ? KV_PRECISIONS[1] : KV_PRECISIONS[2];
        const savedKv = calc.kv * (1 - target.bytes / kvPrecision.bytes);
        out.push({
          tone: 'warning',
          icon: Database,
          title: `Quantize the KV cache to ${target.label}`,
          detail: `The cache is ${fmtGB(calc.kv)} at ${kvPrecision.label}. Dropping to ${target.label} frees ${fmtGB(savedKv)} while keeping the full ${fmtCtx(context)} window.`,
          action: () => setKvPrecisionId(target.id),
          actionLabel: `Use ${target.label} cache`,
        });
      }

      // Reduce context to the largest step that fits.
      const fittingStep = [...CONTEXT_STEPS].reverse().find((c) => c <= calc.maxContext);
      if (fittingStep && fittingStep < context) {
        out.push({
          tone: 'warning',
          icon: Rows3,
          title: `Reduce context to ${fmtCtx(fittingStep)}`,
          detail: `${fmtCtx(context)} needs ${fmtGB(calc.kv)} of cache. ${fmtCtx(fittingStep)} is the largest window that fits in ${fmtGB(capacityGB)} with this model and quantization.`,
          action: () => setCtxIndex(CONTEXT_STEPS.indexOf(fittingStep)),
          actionLabel: `Set ${fmtCtx(fittingStep)}`,
        });
      }

      if (!flashAttention) {
        out.push({
          tone: 'warning',
          icon: Zap,
          title: 'Enable Flash Attention',
          detail: `Fused attention removes the ${fmtGB(calc.attentionScratch)} score buffer and speeds up prompt processing. Supported by llama.cpp (-fa), vLLM, and ExLlamaV2.`,
          action: () => setFlashAttention(true),
          actionLabel: 'Enable',
        });
      }

      if (batchSize > 1) {
        const perStream = calc.kv / batchSize;
        out.push({
          tone: 'warning',
          icon: Layers,
          title: 'Lower the batch size',
          detail: `Each concurrent sequence carries its own ${fmtGB(perStream)} cache. Serving 1 stream instead of ${batchSize} frees ${fmtGB(calc.kv - perStream)}.`,
          action: () => setBatchSize(1),
          actionLabel: 'Set batch 1',
        });
      }

      // When no single change is enough, show the most aggressive
      // configuration that would actually fit on this hardware.
      if (!closing) {
        const smallest = QUANTS.reduce((a, b) => (b.bpw < a.bpw ? b : a));
        const minWeights = weightsFor(smallest);
        const minKvPerToken = (2 * arch.layers * arch.kvHeads * arch.headDim * 0.5) / GIB;
        const bestCtx = [...CONTEXT_STEPS]
          .reverse()
          .find((c) => minWeights + minKvPerToken * c + BASELINE_OVERHEAD_GB <= capacityGB);

        if (bestCtx) {
          const planTotal = minWeights + minKvPerToken * bestCtx + BASELINE_OVERHEAD_GB;
          out.push({
            tone: 'critical',
            icon: SlidersHorizontal,
            title: `Minimum viable config: ${smallest.label} + Q4 cache + ${fmtCtx(bestCtx)}`,
            detail: `The most aggressive combination that fits ${fmtGB(capacityGB)} — ${fmtGB(planTotal)} total, one stream. Quality index drops to ${smallest.quality}, so weigh this against a smaller model at better precision.`,
            action: () => {
              setQuantId(smallest.id);
              setKvPrecisionId('q4');
              setCtxIndex(CONTEXT_STEPS.indexOf(bestCtx));
              setBatchSize(1);
            },
            actionLabel: 'Apply plan',
          });
        } else {
          const cards = Math.ceil((minWeights + minKvPerToken * 2048 + BASELINE_OVERHEAD_GB) / capacityGB);
          out.push({
            tone: 'critical',
            icon: Server,
            title: `${arch.params}B will not fit this device at any setting`,
            detail: `Even ${smallest.label} weights with a Q4 cache at 2k context need more than ${fmtGB(capacityGB)}. Plan on roughly ${cards}× this hardware, or step down to a smaller model.`,
          });
        }
      }

      // Hardware that would take this configuration as-is.
      const fitting = HARDWARE.filter((h) => h.id !== 'custom' && h.vram >= calc.total).sort(
        (a, b) => a.vram - b.vram
      );
      const largest = HARDWARE.filter((h) => h.id !== 'custom').sort((a, b) => b.vram - a.vram)[0];
      if (fitting.length > 0) {
        const upgrade = fitting[0];
        out.push({
          tone: 'info',
          icon: Server,
          title: `Smallest hardware that fits: ${upgrade.label}`,
          detail: `${fmtGB(upgrade.vram)} usable at ${fmtInt(upgrade.bandwidth)} GB/s would run this configuration with ${fmtGB(upgrade.vram - calc.total)} to spare.`,
          action: () => setHardwareId(upgrade.id),
          actionLabel: 'Select',
        });
      } else {
        out.push({
          tone: 'info',
          icon: Server,
          title: 'No listed preset holds this configuration',
          detail: `The largest option here is ${largest.label} at ${fmtGB(largest.vram)}, still ${fmtGB(calc.total - largest.vram)} short. This configuration needs a multi-node split or a lighter footprint.`,
        });
      }

      if (calc.cpuOffloadPossible) {
        out.push({
          tone: 'info',
          icon: HardDrive,
          title: `Offload at least ~${Math.ceil(calc.offloadFraction * 100)}% of weights to CPU`,
          detail: `This can make the memory footprint fit. The resulting theoretical decode ceiling is ≤${fmtTps(calc.tps)} tok/s, versus a fully resident ceiling of ≤${fmtTps(calc.idealTps)} tok/s. These are bandwidth-derived upper bounds, not expected benchmark performance.`,
        });
      } else {
        out.push({
          tone: 'critical',
          icon: HardDrive,
          title: 'CPU offload cannot make this configuration fit',
          detail: `KV cache plus runtime memory require ${fmtGB(calc.nonWeightMemory)}, exceeding ${fmtGB(capacityGB)} before model weights are loaded. Reduce context, batch size, or KV precision, or select a larger device.`,
        });
      }
    } else {
      const headroom = calc.remaining;

      const biggerStep = [...CONTEXT_STEPS].reverse().find((c) => c <= calc.maxContext && c > context);
      if (biggerStep) {
        out.push({
          tone: 'good',
          icon: Rows3,
          title: `Room for a ${fmtCtx(biggerStep)} context window`,
          detail: `${fmtGB(headroom)} is free. Extending from ${fmtCtx(context)} to ${fmtCtx(biggerStep)} still fits inside ${fmtGB(capacityGB)}.`,
          action: () => setCtxIndex(CONTEXT_STEPS.indexOf(biggerStep)),
          actionLabel: `Set ${fmtCtx(biggerStep)}`,
        });
      }

      const better = QUANTS.filter((q) => q.bpw > quant.bpw).sort((a, b) => a.bpw - b.bpw);
      for (const q of better) {
        const newWeights = ((arch.params * 1e9 * (q.bpw / 8)) / GIB) * 1.2;
        if (newWeights + calc.kv + calc.overhead <= capacityGB) {
          out.push({
            tone: 'good',
            icon: Boxes,
            title: `Upgrade quality: ${quant.label} → ${q.label}`,
            detail: `${fmtGB(newWeights)} of weights still fits. Quality index ${quant.quality} → ${q.quality} at a cost of ${fmtGB(newWeights - calc.weights)}.`,
            action: () => setQuantId(q.id),
            actionLabel: `Apply ${q.label}`,
          });
          break;
        }
      }

      if (headroom > 2 && batchSize === 1) {
        const extraStreams = Math.floor(headroom / Math.max(calc.kv, 0.001));
        if (extraStreams >= 1) {
          out.push({
            tone: 'good',
            icon: Layers,
            title: `Serve up to ${Math.min(64, extraStreams + 1)} concurrent streams`,
            detail: `Free memory covers ${extraStreams} more ${fmtCtx(context)} cache${extraStreams === 1 ? '' : 's'}. Batched decode reuses one weight read per step, so the modeled aggregate theoretical ceiling scales sub-linearly.`,
          });
        }
      }

      if (!flashAttention) {
        out.push({
          tone: 'info',
          icon: Zap,
          title: 'Enable Flash Attention anyway',
          detail: 'Fits without it, but fused attention still cuts prompt-processing latency and the scratch buffer.',
          action: () => setFlashAttention(true),
          actionLabel: 'Enable',
        });
      }

      if (headroom < capacityGB * 0.08) {
        out.push({
          tone: 'warning',
          icon: AlertTriangle,
          title: 'Headroom is thin',
          detail: `Only ${fmtGB(headroom)} spare. A desktop compositor, a second CUDA process, or a long prompt burst can push this configuration into an out-of-memory error.`,
        });
      }
    }

    return out;
  }, [
    calc,
    quant,
    arch,
    kvPrecision,
    kvPrecisionId,
    context,
    batchSize,
    flashAttention,
    capacityGB,
    bandwidthGBPerSecond,
  ]);

  /* ---------------- feasibility table ---------------- */
  const tableRows = [
    {
      component: 'Model weights',
      color: T.weights,
      detail: `${arch.params}B @ ${quant.label} (${quant.bpw} bpw) × 1.2 overhead`,
      size: calc.weights,
    },
    {
      component: 'KV cache',
      color: T.kv,
      detail: `${fmtCtx(context)} ctx × ${batchSize} seq × ${arch.layers} layers @ ${kvPrecision.label}`,
      size: calc.kv,
    },
    {
      component: 'Runtime overhead',
      color: T.overhead,
      detail: flashAttention
        ? `${fmtGB(BASELINE_OVERHEAD_GB)} CUDA/runtime baseline`
        : `${fmtGB(BASELINE_OVERHEAD_GB)} baseline + ${fmtGB(calc.attentionScratch)} attention scratch`,
      size: calc.overhead,
    },
  ];

  const statusTone = !calc.fits ? 'critical' : calc.utilisation > 92 ? 'warning' : 'good';
  const statusColor =
    statusTone === 'critical' ? T.critical : statusTone === 'warning' ? T.warning : T.good;
  const StatusIcon = statusTone === 'critical' ? XCircle : statusTone === 'warning' ? AlertTriangle : CheckCircle2;
  const statusLabel = !calc.fits ? 'FAIL — over capacity' : calc.utilisation > 92 ? 'PASS — tight fit' : 'PASS — fits';

  const toneStyles = {
    critical: { color: T.critical, bg: 'rgba(208,59,59,0.12)' },
    warning: { color: T.warning, bg: 'rgba(250,178,25,0.12)' },
    good: { color: T.good, bg: 'rgba(12,163,12,0.12)' },
    info: { color: T.weights, bg: 'rgba(57,135,229,0.12)' },
  };

  return (
    <div className="min-h-screen bg-[#0a0a0c] text-[#f2f2f5]">
      {/* ---------------- header ---------------- */}
      <header className="border-b" style={{ borderColor: T.border }}>
        <div className="mx-auto flex max-w-[1400px] flex-col gap-3 px-4 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-6">
          <div className="flex items-center gap-3">
            <span
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border"
              style={{ borderColor: T.border, background: T.surface }}
            >
              <Terminal size={17} strokeWidth={2} style={{ color: T.weights }} aria-hidden="true" />
            </span>
            <div>
              <h1 className="font-mono text-sm font-semibold tracking-tight text-[#f2f2f5] sm:text-base">
                VRAM Calc <span className="text-[#8a8a99]">&amp;</span> Local LLM Profiler
              </h1>
              <p className="mt-0.5 text-[11px] text-[#8a8a99]">
                Hardware feasibility for local Llama · Qwen · DeepSeek · Mistral
              </p>
            </div>
          </div>
          <div
            className="flex items-center gap-2 self-start rounded-lg px-3 py-2 sm:self-auto"
            style={{ background: toneStyles[statusTone].bg }}
          >
            <StatusIcon size={15} strokeWidth={2.2} style={{ color: statusColor }} aria-hidden="true" />
            <span className="font-mono text-xs font-semibold" style={{ color: statusColor }}>
              {statusLabel}
            </span>
            <span className="font-mono text-xs text-[#b4b4c2] tabular-nums">
              {calc.utilisation.toFixed(0)}%
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1400px] px-4 py-5 sm:px-6 sm:py-6">
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-[360px_minmax(0,1fr)] xl:grid-cols-[380px_minmax(0,1fr)]">
          {/* ============ configuration column ============ */}
          <div className="space-y-5">
            <Panel title="Model" icon={Cpu}>
              <div className="space-y-4">
                <Field label="Parameters" hint={`${arch.params}B`} htmlFor="model-size">
                  <Segmented
                    ariaLabel="Model parameter size"
                    columns={3}
                    value={modelId}
                    onChange={setModelId}
                    options={MODELS.map((m) => ({ value: m.id, label: m.label }))}
                  />
                  <p className="pt-1 text-[11px] leading-snug text-[#8a8a99]">{model.example}</p>
                </Field>

                {modelId === 'custom' && (
                  <Field label="Custom parameter count" hint="billions" htmlFor="custom-params">
                    <NumberInput
                      id="custom-params"
                      value={customParams}
                      onChange={setCustomParams}
                      min={0.5}
                      max={1000}
                      step={0.5}
                      suffix="B"
                    />
                  </Field>
                )}

                <Field label="Quantization" hint={`${quant.bpw} bits/weight`} htmlFor="quant">
                  <Segmented
                    ariaLabel="Quantization format"
                    columns={3}
                    value={quantId}
                    onChange={setQuantId}
                    options={QUANTS.map((q) => ({ value: q.id, label: q.label }))}
                  />
                  <p className="pt-1 text-[11px] leading-snug text-[#8a8a99]">{quant.note}</p>
                </Field>

                <div
                  className="flex items-center justify-between rounded-lg border px-3 py-2"
                  style={{ borderColor: T.border, background: T.surfaceInput }}
                >
                  <span className="font-mono text-[11px] uppercase tracking-[0.12em] text-[#8a8a99]">
                    Architecture
                  </span>
                  <span className="font-mono text-[11px] text-[#b4b4c2] tabular-nums">
                    {arch.layers}L · {arch.hidden}d · {arch.kvHeads} KV heads
                    {arch.derived && <span className="text-[#8a8a99]"> (est.)</span>}
                  </span>
                </div>
              </div>
            </Panel>

            <Panel title="Inference" icon={SlidersHorizontal}>
              <div className="space-y-4">
                <Field label="Context window" hint={`${fmtCtx(context)} · ${fmtInt(context)} tokens`} htmlFor="ctx">
                  <Slider
                    id="ctx"
                    min={0}
                    max={CONTEXT_STEPS.length - 1}
                    value={ctxIndex}
                    onChange={setCtxIndex}
                  />
                  <div className="flex justify-between pt-1 font-mono text-[10px] text-[#8a8a99]">
                    <span>2k</span>
                    <span>32k</span>
                    <span>128k</span>
                  </div>
                </Field>

                <Field label="Batch size" hint={`${batchSize} concurrent`} htmlFor="batch">
                  <div className="flex items-center gap-3">
                    <Slider id="batch" min={1} max={64} value={batchSize} onChange={setBatchSize} />
                    <div className="w-20 shrink-0">
                      <NumberInput value={batchSize} onChange={setBatchSize} min={1} max={64} />
                    </div>
                  </div>
                </Field>

                <Field label="KV cache precision" hint={`${kvPrecision.bytes} B/element`} htmlFor="kvp">
                  <Segmented
                    ariaLabel="KV cache precision"
                    columns={3}
                    value={kvPrecisionId}
                    onChange={setKvPrecisionId}
                    options={KV_PRECISIONS.map((k) => ({ value: k.id, label: k.label }))}
                  />
                </Field>

                <Toggle
                  id="flash-attention"
                  checked={flashAttention}
                  onChange={setFlashAttention}
                  label="Flash Attention"
                  description="Fused attention kernel — removes the materialised score buffer."
                />
              </div>
            </Panel>

            <Panel title="Hardware" icon={Server}>
              <div className="space-y-4">
                <Field label="Preset" htmlFor="hardware">
                  <Select
                    id="hardware"
                    value={hardwareId}
                    onChange={setHardwareId}
                    options={HARDWARE.map((h) => ({ value: h.id, label: h.label }))}
                  />
                  <p className="pt-1 text-[11px] leading-snug text-[#8a8a99]">{hardwarePreset.note}</p>
                </Field>

                {hardwareId === 'custom' ? (
                  <>
                    <Field label="Usable VRAM" hint={fmtGB(customVram)} htmlFor="cvram">
                      <Slider id="cvram" min={4} max={256} step={2} value={customVram} onChange={setCustomVram} />
                    </Field>
                    <Field label="Memory bandwidth" hint={`${fmtInt(customBandwidth)} GB/s`} htmlFor="cbw">
                      <Slider
                        id="cbw"
                        min={50}
                        max={4000}
                        step={25}
                        value={customBandwidth}
                        onChange={setCustomBandwidth}
                      />
                    </Field>
                  </>
                ) : (
                  <div className="grid grid-cols-2 gap-2">
                    <div
                      className="rounded-lg border px-3 py-2.5"
                      style={{ borderColor: T.border, background: T.surfaceInput }}
                    >
                      <div className="font-mono text-[10px] uppercase tracking-[0.1em] text-[#8a8a99]">
                        {capacityLabel}
                      </div>
                      <div className="mt-1 font-mono text-sm text-[#f2f2f5] tabular-nums">
                        {fmtGB(capacityGB)}
                      </div>
                    </div>
                    <div
                      className="rounded-lg border px-3 py-2.5"
                      style={{ borderColor: T.border, background: T.surfaceInput }}
                    >
                      <div className="font-mono text-[10px] uppercase tracking-[0.1em] text-[#8a8a99]">
                        Bandwidth
                      </div>
                      <div className="mt-1 font-mono text-sm text-[#f2f2f5] tabular-nums">
                        {fmtInt(bandwidthGBPerSecond)} GB/s
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </Panel>
          </div>

          {/* ============ dashboard column ============ */}
          <div className="space-y-5">
            {/* metric cards */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <MetricCard
                hero
                icon={MemoryStick}
                label="Total VRAM required"
                value={calc.total.toFixed(1)}
                unit="GB"
                tone={statusTone}
                sub={`${fmtGB(capacityGB)} available · ${calc.utilisation.toFixed(0)}% utilised`}
              />
              <MetricCard
                icon={Gauge}
                label={calc.remaining >= 0 ? 'Free VRAM' : 'VRAM deficit'}
                value={Math.abs(calc.remaining).toFixed(1)}
                unit="GB"
                tone={calc.remaining < 0 ? 'critical' : calc.remaining < capacityGB * 0.08 ? 'warning' : 'good'}
                sub={
                  calc.remaining >= 0
                    ? `Headroom for prompt bursts and a second process`
                    : calc.cpuOffloadPossible
                      ? `Over budget — at least ~${Math.ceil(calc.offloadFraction * 100)}% of weights must be offloaded`
                      : `KV cache + runtime use ${fmtGB(calc.nonWeightMemory)} — CPU offload cannot make it fit`
                }
              />
              <MetricCard
                icon={TrendingUp}
                label="Theoretical decode ceiling"
                value={calc.tps === null ? 'N/A' : `≤${fmtTps(calc.tps)}`}
                unit={calc.tps === null ? undefined : 'tok/s'}
                tone={calc.tps === null ? 'critical' : calc.fits ? 'neutral' : 'warning'}
                sub={
                  calc.fits
                    ? batchSize > 1
                      ? `Bandwidth-derived upper bound; aggregate ceiling ≤${fmtTps(calc.aggregateTps)} tok/s across ${batchSize} streams. Not expected benchmark performance.`
                      : `${fmtInt(bandwidthGBPerSecond)} decimal GB/s converts to ${calc.bandwidthGiBPerSecond.toFixed(1)} GiB/s. Upper bound only, not expected benchmark performance.`
                    : calc.cpuOffloadPossible
                      ? `CPU-offload bandwidth-derived upper bound; fully resident ceiling ≤${fmtTps(calc.idealTps)} tok/s. Not expected benchmark performance.`
                      : `KV cache plus runtime memory exceed device capacity before weights are loaded.`
                }
              />
            </div>

            {/* capacity meter + stacked breakdown */}
            <Panel
              title="VRAM allocation"
              icon={Activity}
              action={
                <span className="font-mono text-[11px] text-[#8a8a99] tabular-nums">
                  {fmtGB(calc.total)} / {fmtGB(capacityGB)}
                </span>
              }
            >
              {/* pass/fail meter */}
              <div className="mb-5">
                <div className="mb-2 flex items-center justify-between">
                  <span className="font-mono text-[11px] uppercase tracking-[0.12em] text-[#8a8a99]">
                    Capacity
                  </span>
                  <span className="font-mono text-[11px] tabular-nums" style={{ color: statusColor }}>
                    {calc.utilisation.toFixed(1)}% of {hardwarePreset.label}
                  </span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full" style={{ background: '#23232c' }}>
                  <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{
                      width: `${Math.min(100, calc.utilisation)}%`,
                      background: statusColor,
                    }}
                  />
                </div>
              </div>

              {/* stacked bar */}
              <div className="h-[128px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    layout="vertical"
                    data={stackData}
                    margin={{ top: 24, right: 16, bottom: 4, left: 14 }}
                    barCategoryGap="30%"
                  >
                    <CartesianGrid horizontal={false} stroke={T.grid} strokeWidth={1} />
                    <XAxis
                      type="number"
                      domain={[0, axisMax]}
                      ticks={niceTicks(axisMax)}
                      tickFormatter={(v) => `${Math.round(v)} GB`}
                      stroke={T.axis}
                      tickLine={false}
                      tick={{ fill: T.inkMuted, fontSize: 11 }}
                      interval={0}
                    />
                    <YAxis type="category" dataKey="name" hide />
                    <Tooltip
                      cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                      content={<ChartTooltip labelText="Allocation" />}
                    />
                    {/* 2px stroke in the surface colour is the gap between
                        touching segments, not a border on the mark. */}
                    <Bar
                      dataKey="weights"
                      name="Model weights"
                      stackId="vram"
                      fill={T.weights}
                      stroke={T.surface}
                      strokeWidth={2}
                      barSize={24}
                      isAnimationActive={false}
                    />
                    <Bar
                      dataKey="kv"
                      name="KV cache"
                      stackId="vram"
                      fill={T.kv}
                      stroke={T.surface}
                      strokeWidth={2}
                      barSize={24}
                      isAnimationActive={false}
                    />
                    <Bar
                      dataKey="overhead"
                      name="Runtime overhead"
                      stackId="vram"
                      fill={T.overhead}
                      stroke={T.surface}
                      strokeWidth={2}
                      barSize={24}
                      radius={[0, 4, 4, 0]}
                      isAnimationActive={false}
                    />
                    <ReferenceLine
                      x={capacityGB}
                      stroke={statusColor}
                      strokeWidth={2}
                      /* Short label so it never runs off the plot when the
                         line sits near the right edge on narrow screens —
                         the meter above names the device in full. */
                      label={{
                        value: fmtGB(capacityGB),
                        position: 'top',
                        fill: statusColor,
                        fontSize: 11,
                        offset: 10,
                      }}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2 border-t pt-3" style={{ borderColor: T.border }}>
                <LegendSwatch color={T.weights} label="Model weights" value={fmtGB(calc.weights)} />
                <LegendSwatch color={T.kv} label="KV cache" value={fmtGB(calc.kv)} />
                <LegendSwatch color={T.overhead} label="Runtime overhead" value={fmtGB(calc.overhead)} />
              </div>
            </Panel>

            {/* context scaling */}
            <Panel
              title="Total VRAM vs. context length"
              icon={Rows3}
              action={
                <span className="font-mono text-[11px] text-[#8a8a99] tabular-nums">
                  {calc.maxContext >= 1
                    ? `max ${fmtInt(Math.min(calc.maxContext, 1e9))} tok`
                    : 'weights alone exceed capacity'}
                </span>
              }
            >
              <div className="h-[190px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={contextCurve} margin={{ top: 20, right: 16, bottom: 0, left: -8 }}>
                    <defs>
                      <linearGradient id="ctxFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={T.weights} stopOpacity={0.18} />
                        <stop offset="100%" stopColor={T.weights} stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke={T.grid} strokeWidth={1} vertical={false} />
                    <XAxis
                      dataKey="label"
                      stroke={T.axis}
                      tickLine={false}
                      tick={{ fill: T.inkMuted, fontSize: 11 }}
                    />
                    <YAxis
                      stroke={T.axis}
                      tickLine={false}
                      axisLine={false}
                      width={68}
                      tick={{ fill: T.inkMuted, fontSize: 11 }}
                      ticks={niceTicks(curveMax)}
                      domain={[0, curveMax]}
                      tickFormatter={(v) => `${Math.round(v)} GB`}
                    />
                    <Tooltip
                      cursor={{ stroke: T.axis, strokeWidth: 1 }}
                      content={({ active, payload, label }) => (
                        <ChartTooltip active={active} payload={payload} labelText={`${label} context`} />
                      )}
                    />
                    <Area
                      type="monotone"
                      dataKey="total"
                      name="Total required"
                      stroke={T.weights}
                      strokeWidth={2}
                      fill="url(#ctxFill)"
                      dot={false}
                      activeDot={{ r: 4, fill: T.weights, stroke: T.surface, strokeWidth: 2 }}
                      isAnimationActive={false}
                    />
                    <ReferenceLine
                      y={capacityGB}
                      stroke={statusColor}
                      strokeWidth={2}
                      strokeDasharray="0"
                      label={{
                        value: `capacity ${fmtGB(capacityGB)}`,
                        position: 'insideTopLeft',
                        fill: statusColor,
                        fontSize: 11,
                      }}
                    />
                    <ReferenceDot
                      x={fmtCtx(context)}
                      y={Number(calc.total.toFixed(3))}
                      r={5}
                      fill={T.weights}
                      stroke={T.surface}
                      strokeWidth={2}
                      isFront
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <p className="mt-2 text-[11px] leading-snug text-[#8a8a99]">
                Weights are fixed; the slope is the KV cache at {batchSize} concurrent sequence
                {batchSize === 1 ? '' : 's'}. The marked point is the current {fmtCtx(context)} configuration.
              </p>
            </Panel>

            {/* recommendations */}
            <Panel
              title={calc.fits ? 'Headroom & tuning' : 'Fix this configuration'}
              icon={calc.fits ? Lightbulb : AlertTriangle}
              action={
                <span className="font-mono text-[11px] text-[#8a8a99] tabular-nums">
                  {recommendations.length} suggestion{recommendations.length === 1 ? '' : 's'}
                </span>
              }
            >
              {recommendations.length === 0 ? (
                <p className="text-sm text-[#b4b4c2]">
                  This configuration is balanced — nothing worth changing at these settings.
                </p>
              ) : (
                <ul className="space-y-2.5">
                  {recommendations.map((rec, i) => {
                    const style = toneStyles[rec.tone];
                    const Icon = rec.icon;
                    return (
                      <li
                        key={i}
                        className="flex flex-col gap-3 rounded-lg border p-3 sm:flex-row sm:items-start"
                        style={{ borderColor: T.border, background: T.surfaceInput }}
                      >
                        <span
                          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md"
                          style={{ background: style.bg }}
                        >
                          <Icon size={14} strokeWidth={2.2} style={{ color: style.color }} aria-hidden="true" />
                        </span>
                        <div className="min-w-0 flex-1">
                          <p className="font-mono text-xs font-semibold text-[#f2f2f5]">{rec.title}</p>
                          <p className="mt-1 text-[12px] leading-relaxed text-[#b4b4c2]">{rec.detail}</p>
                        </div>
                        {rec.action && (
                          <button
                            type="button"
                            onClick={rec.action}
                            className="shrink-0 self-start rounded-md border px-2.5 py-1.5 font-mono text-[11px] text-[#f2f2f5] transition hover:border-[rgba(255,255,255,0.28)] hover:bg-[#1c1c24] focus:outline-none focus-visible:ring-1 focus-visible:ring-[#3987e5]"
                            style={{ borderColor: T.borderStrong }}
                          >
                            {rec.actionLabel}
                          </button>
                        )}
                      </li>
                    );
                  })}
                </ul>
              )}
            </Panel>

            {/* feasibility table */}
            <Panel title="Feasibility breakdown" icon={Rows3}>
              <div className="-mx-4 overflow-x-auto px-4">
                <table className="w-full min-w-[560px] border-collapse text-left">
                  <thead>
                    <tr className="border-b" style={{ borderColor: T.border }}>
                      {['Component', 'Derivation', 'Size', 'Share'].map((h, i) => (
                        <th
                          key={h}
                          className={`pb-2 font-mono text-[10px] uppercase tracking-[0.12em] text-[#8a8a99] ${
                            i >= 2 ? 'text-right' : ''
                          }`}
                        >
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {tableRows.map((row) => (
                      <tr key={row.component} className="border-b" style={{ borderColor: T.border }}>
                        <td className="py-2.5 pr-4">
                          <span className="flex items-center gap-2">
                            <span
                              className="h-2.5 w-2.5 shrink-0 rounded-sm"
                              style={{ background: row.color }}
                              aria-hidden="true"
                            />
                            <span className="font-mono text-xs text-[#f2f2f5]">{row.component}</span>
                          </span>
                        </td>
                        <td className="py-2.5 pr-4 text-[11px] text-[#8a8a99]">{row.detail}</td>
                        <td className="py-2.5 pr-4 text-right font-mono text-xs text-[#f2f2f5] tabular-nums">
                          {fmtGB(row.size)}
                        </td>
                        <td className="py-2.5 text-right font-mono text-xs text-[#b4b4c2] tabular-nums">
                          {((row.size / calc.total) * 100).toFixed(0)}%
                        </td>
                      </tr>
                    ))}
                    <tr className="border-b" style={{ borderColor: T.borderStrong }}>
                      <td className="py-2.5 pr-4 font-mono text-xs font-semibold text-[#f2f2f5]">
                        Total required
                      </td>
                      <td className="py-2.5 pr-4 text-[11px] text-[#8a8a99]">
                        weights + KV cache + runtime overhead
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono text-xs font-semibold text-[#f2f2f5] tabular-nums">
                        {fmtGB(calc.total)}
                      </td>
                      <td className="py-2.5 text-right font-mono text-xs text-[#b4b4c2] tabular-nums">100%</td>
                    </tr>
                    <tr className="border-b" style={{ borderColor: T.border }}>
                      <td className="py-2.5 pr-4 font-mono text-xs text-[#f2f2f5]">Hardware capacity</td>
                      <td className="py-2.5 pr-4 text-[11px] text-[#8a8a99]">
                        {hardwarePreset.label} @ {fmtInt(bandwidthGBPerSecond)} GB/s
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono text-xs text-[#f2f2f5] tabular-nums">
                        {fmtGB(capacityGB)}
                      </td>
                      <td className="py-2.5 text-right font-mono text-xs text-[#b4b4c2] tabular-nums">
                        {calc.utilisation.toFixed(0)}% used
                      </td>
                    </tr>
                    <tr>
                      <td className="py-2.5 pr-4 font-mono text-xs text-[#f2f2f5]">Verdict</td>
                      <td className="py-2.5 pr-4 text-[11px] text-[#8a8a99]">
                        {calc.fits
                          ? `Fully resident on device · theoretical decode ceiling ≤${fmtTps(calc.tps)} tok/s`
                          : calc.cpuOffloadPossible
                            ? `Short by ${fmtGB(Math.abs(calc.remaining))} · offload at least ~${Math.ceil(calc.offloadFraction * 100)}% of weights · theoretical ceiling ≤${fmtTps(calc.tps)} tok/s`
                            : `KV cache + runtime require ${fmtGB(calc.nonWeightMemory)} · CPU offload cannot make this configuration fit`}
                      </td>
                      <td colSpan={2} className="py-2.5 text-right">
                        <span
                          className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 font-mono text-[11px] font-semibold"
                          style={{ background: toneStyles[statusTone].bg, color: statusColor }}
                        >
                          <StatusIcon size={12} strokeWidth={2.4} aria-hidden="true" />
                          {calc.fits ? 'PASS' : 'FAIL'}
                        </span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </Panel>

            {/* assumptions */}
            <div
              className="rounded-xl border p-4"
              style={{ borderColor: T.border, background: T.surface }}
            >
              <div className="flex items-center gap-2">
                <Info size={14} strokeWidth={2} style={{ color: T.inkMuted }} aria-hidden="true" />
                <h2 className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#b4b4c2]">
                  Model assumptions
                </h2>
              </div>
              <ul className="mt-3 grid gap-2 text-[11px] leading-relaxed text-[#8a8a99] sm:grid-cols-2">
                <li>
                  <span className="text-[#b4b4c2]">Weights</span> = params × (bpw ÷ 8) × 1.2, where 1.2 covers
                  allocator padding and fragmentation on load.
                </li>
                <li>
                  <span className="text-[#b4b4c2]">KV cache</span> = 2 × layers × kv-heads × head-dim × context ×
                  batch × bytes-per-element. Grouped-query attention is assumed (8 KV heads).
                </li>
                <li>
                  <span className="text-[#b4b4c2]">Overhead</span> = 1.5 GB CUDA/runtime baseline, plus 25% of the
                  cache as attention scratch when Flash Attention is off.
                </li>
                <li>
                  <span className="text-[#b4b4c2]">Theoretical decode ceiling</span> = bandwidth in GiB/s ÷ weight
                  footprint in GiB. Manufacturer decimal GB/s is converted to GiB/s first. This upper bound is not
                  expected benchmark performance.
                </li>
                <li>
                  <span className="text-[#b4b4c2]">Offload</span> assumes {SYSTEM_RAM_BANDWIDTH_GB_PER_SECOND} GB/s of system
                  memory bandwidth for the weights that spill out of VRAM. It is viable only when KV cache plus
                  runtime memory fit on the device without model weights.
                </li>
                <li>
                  <span className="text-[#b4b4c2]">GB</span> means GiB (2<sup>30</sup> bytes). Unified-memory
                  presets list the practically allocatable share, not total system memory.
                </li>
              </ul>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
