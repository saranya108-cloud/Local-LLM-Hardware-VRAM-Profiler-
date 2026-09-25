import test from 'node:test';
import assert from 'node:assert/strict';

import { BASELINE_OVERHEAD_GB, SYSTEM_RAM_BANDWIDTH_GB_PER_SECOND } from '../src/calculations.js';
import { BACKENDS, DEFAULT_BACKEND_ID, preferredQuantId, quantsForBackend } from '../src/data/backends.js';
import { DEFAULT_HARDWARE_ID, HARDWARE } from '../src/data/hardware.js';
import { DEFAULT_MARGIN_ID, MARGIN_BUCKETS, MARGIN_PROFILES, sumMarginBuckets } from '../src/data/margins.js';
import { DEFAULT_MEMORY_ID, MEMORY_PROFILES, selectableMemoryProfiles } from '../src/data/memory.js';
import { MODELS } from '../src/data/models.js';
import { KV_PRECISIONS, QUANTS } from '../src/data/quantization.js';

function assertUniqueIds(records, label) {
  const ids = records.map((record) => record.id);
  const duplicates = ids.filter((id, index) => ids.indexOf(id) !== index);
  assert.deepEqual(duplicates, [], `${label} ids must be unique`);
  for (const id of ids) {
    assert.equal(typeof id, 'string');
    assert.ok(id.length > 0, `${label} ids must be non-empty`);
  }
}

function assertPositiveFinite(value, label) {
  assert.equal(Number.isFinite(value), true, `${label} must be finite`);
  assert.ok(value > 0, `${label} must be positive`);
}

const SUPPORTED_QUANTS = {
  fp16: 16,
  q8_0: 8.5,
  q6_k: 6.56,
  q4_k_m: 4.85,
  q3_k_s: 3.44,
  exl2: 4.25,
};

const SUPPORTED_KV_PRECISIONS = {
  fp16: 2,
  q8: 1,
  q4: 0.5,
};

test('hardware presets have unique ids and positive capacity and bandwidth', () => {
  assert.ok(HARDWARE.length > 0);
  assertUniqueIds(HARDWARE, 'hardware');

  for (const preset of HARDWARE) {
    assertPositiveFinite(preset.vram, `${preset.id} vram`);
    assertPositiveFinite(preset.bandwidth, `${preset.id} bandwidth`);
  }
});

test('the default hardware preset is the RTX 4090', () => {
  assert.equal(DEFAULT_HARDWARE_ID, 'rtx4090');

  const preset = HARDWARE.find((item) => item.id === DEFAULT_HARDWARE_ID);
  assert.ok(preset, 'default hardware id must exist in the catalog');
  assert.equal(preset.label, 'RTX 4090 24GB');
  assert.equal(preset.vram, 24);
  assert.equal(preset.bandwidth, 1008);
  assert.equal(preset.kind, 'gpu');
});

test('the GB10 preset uses published unified-memory capacity and bandwidth', () => {
  const preset = HARDWARE.find((item) => item.id === 'gb10');
  assert.ok(preset, 'GB10 preset must be present');
  assert.equal(preset.vram, 128);
  assert.equal(preset.bandwidth, 273);
  assert.equal(preset.kind, 'unified');
});

test('models have positive parameter counts and unique ids', () => {
  assert.ok(MODELS.length > 0);
  assertUniqueIds(MODELS, 'model');

  for (const model of MODELS) {
    assertPositiveFinite(model.params, `${model.id} params`);
  }
});

test('quantization formats have valid bits per weight', () => {
  assertUniqueIds(QUANTS, 'quantization');

  for (const quant of QUANTS) {
    assertPositiveFinite(quant.bpw, `${quant.id} bpw`);
    assert.ok(quant.bpw <= 16, `${quant.id} bpw must not exceed FP16`);
  }
});

test('quantization and KV catalogs match the supported formats', () => {
  assert.deepEqual(
    Object.fromEntries(QUANTS.map((quant) => [quant.id, quant.bpw])),
    SUPPORTED_QUANTS,
  );

  assertUniqueIds(KV_PRECISIONS, 'KV precision');
  for (const precision of KV_PRECISIONS) {
    assertPositiveFinite(precision.bytes, `${precision.id} bytes`);
  }
  assert.deepEqual(
    Object.fromEntries(KV_PRECISIONS.map((precision) => [precision.id, precision.bytes])),
    SUPPORTED_KV_PRECISIONS,
  );
});

test('selectable system-memory profiles are generic DDR at 60 GB/s and custom', () => {
  assert.equal(DEFAULT_MEMORY_ID, 'generic-ddr');
  assertUniqueIds(MEMORY_PROFILES, 'memory');

  const selectable = selectableMemoryProfiles();
  assert.deepEqual(
    selectable.map((profile) => profile.id),
    ['generic-ddr', 'custom'],
  );

  const generic = MEMORY_PROFILES.find((profile) => profile.id === 'generic-ddr');
  assert.equal(generic.mode, 'spill');
  assert.equal(generic.bandwidthGBPerSecond, 60);
  assert.equal(generic.bandwidthGBPerSecond, SYSTEM_RAM_BANDWIDTH_GB_PER_SECOND);

  const custom = MEMORY_PROFILES.find((profile) => profile.id === 'custom');
  assert.equal(custom.mode, 'spill');
  assert.equal('bandwidthGBPerSecond' in custom, false);

  for (const preset of HARDWARE.filter((item) => item.kind === 'unified')) {
    assert.notEqual(generic.bandwidthGBPerSecond, preset.bandwidth);
  }
});

test('DDR5 desktop and server memory stay unnamed until a source exists', () => {
  for (const id of ['ddr5-desktop', 'ddr5-server']) {
    const profile = MEMORY_PROFILES.find((item) => item.id === id);
    assert.ok(profile, `${id} must stay reserved`);
    assert.equal(profile.selectable, false);
    assert.equal(profile.mode, 'spill');
    assert.equal('bandwidthGBPerSecond' in profile, false);
  }

  assert.equal(
    selectableMemoryProfiles().some((profile) => profile.id.startsWith('ddr5-')),
    false,
  );
});

test('default margin buckets sum to the 1.5 GB baseline', () => {
  assert.equal(DEFAULT_MARGIN_ID, 'default');
  assertUniqueIds(MARGIN_BUCKETS, 'margin bucket');

  const gb = Object.fromEntries(MARGIN_BUCKETS.map((bucket) => [bucket.id, bucket.gb]));
  assert.deepEqual(Object.keys(gb).sort(), ['baseline', 'driver', 'framework', 'os', 'peaks']);
  assert.equal(gb.baseline, 1.5);
  assert.equal(gb.os, 0);
  assert.equal(gb.driver, 0);
  assert.equal(gb.framework, 0);
  assert.equal(gb.peaks, 0);
  assert.equal(gb.baseline + gb.os + gb.driver + gb.framework + gb.peaks, 1.5);
  assert.equal(gb.baseline + gb.os + gb.driver + gb.framework + gb.peaks, BASELINE_OVERHEAD_GB);

  const profile = MARGIN_PROFILES.find((item) => item.id === DEFAULT_MARGIN_ID);
  assert.deepEqual(profile.buckets, gb);
  assert.equal(sumMarginBuckets(profile.buckets), BASELINE_OVERHEAD_GB);
  assert.equal(
    MARGIN_BUCKETS.filter((bucket) => bucket.modeled).map((bucket) => bucket.id).join(','),
    'baseline',
  );
});

test('backend profiles keep generic math and only hide offload where specified', () => {
  assert.equal(DEFAULT_BACKEND_ID, 'generic');
  assertUniqueIds(BACKENDS, 'backend');

  const byId = Object.fromEntries(BACKENDS.map((backend) => [backend.id, backend]));
  assert.deepEqual(Object.keys(byId).sort(), [
    'exllamav2',
    'generic',
    'llama.cpp',
    'mlx',
    'ollama',
    'vllm',
  ]);

  assert.equal(byId.generic.quantFamily, 'all');
  assert.equal(byId.generic.cpuOffload, 'weights-if-kv-fits');
  assert.equal(byId['llama.cpp'].quantFamily, 'gguf');
  assert.equal(byId['llama.cpp'].cpuOffload, 'weights-if-kv-fits');
  assert.equal(byId.ollama.quantFamily, 'gguf');
  assert.equal(byId.ollama.cpuOffload, 'weights-if-kv-fits');
  assert.equal(byId.mlx.quantFamily, 'mlx');
  assert.equal(byId.mlx.cpuOffload, 'weights-if-kv-fits');
  assert.equal(byId.exllamav2.quantFamily, 'exl2');
  assert.equal(byId.exllamav2.cpuOffload, 'disabled');
  assert.equal(byId.vllm.quantFamily, 'gguf');
  assert.equal(byId.vllm.cpuOffload, 'disabled');

  for (const backend of BACKENDS) {
    assert.equal('overheadGB' in backend, false);
    assert.equal('bandwidthScale' in backend, false);
  }

  const ggufIds = QUANTS.filter((quant) => quant.family === 'gguf').map((quant) => quant.id);
  assert.deepEqual(quantsForBackend(byId['llama.cpp']).map((quant) => quant.id), ggufIds);
  assert.deepEqual(quantsForBackend(byId.ollama).map((quant) => quant.id), ggufIds);
  assert.deepEqual(quantsForBackend(byId.exllamav2).map((quant) => quant.id), ['exl2']);
  assert.deepEqual(quantsForBackend(byId.mlx).map((quant) => quant.id), []);
  assert.equal(quantsForBackend(byId.generic).length, QUANTS.length);
  assert.deepEqual(quantsForBackend(byId.vllm).map((quant) => quant.id), ggufIds);
  assert.equal(quantsForBackend(byId.vllm).some((quant) => quant.id === 'exl2'), false);
  assert.equal(QUANTS.some((quant) => quant.family === 'mlx'), false);

  assert.equal(preferredQuantId(byId['llama.cpp'], 'exl2'), 'q4_k_m');
  assert.equal(preferredQuantId(byId['llama.cpp'], 'q6_k'), 'q6_k');
  assert.equal(preferredQuantId(byId.ollama, 'exl2'), 'q4_k_m');
  assert.equal(preferredQuantId(byId.exllamav2, 'q4_k_m'), 'exl2');
  assert.equal(preferredQuantId(byId.generic, 'exl2'), 'exl2');
  assert.equal(preferredQuantId(byId.vllm, 'q4_k_m'), 'q4_k_m');
  assert.equal(preferredQuantId(byId.vllm, 'exl2'), 'q4_k_m');
  assert.equal(preferredQuantId(byId.mlx, 'q4_k_m'), null);
});
