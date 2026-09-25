import test from 'node:test';
import assert from 'node:assert/strict';

import { DEFAULT_HARDWARE_ID, HARDWARE } from '../src/data/hardware.js';
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
