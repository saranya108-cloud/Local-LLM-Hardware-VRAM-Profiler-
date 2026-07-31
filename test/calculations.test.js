import test from 'node:test';
import assert from 'node:assert/strict';

import { calculateProfile, decimalGBToGiB } from '../src/calculations.js';

function approx(actual, expected, tolerance = 1e-10) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `expected ${actual} to be within ${tolerance} of ${expected}`
  );
}

const arch = {
  params: 8,
  layers: 32,
  kvHeads: 8,
  headDim: 128,
};

const expectedWeightsRaw = (8 * 1e9 * (4.85 / 8)) / (1024 ** 3);
const expectedWeights = expectedWeightsRaw * 1.2;
const expectedKvPerTokenGB = (2 * 32 * 8 * 128 * 2) / (1024 ** 3);
const expectedKv = expectedKvPerTokenGB * 8192 * 1;
const expectedTotal = expectedWeights + expectedKv + 1.5;
const expectedMaxContext = (24 - expectedWeights - 1.5) / expectedKvPerTokenGB;

function profile(overrides = {}) {
  return calculateProfile({
    arch,
    bitsPerWeight: 4.85,
    kvBytesPerElement: 2,
    context: 8192,
    batchSize: 1,
    flashAttention: true,
    capacityGB: 24,
    bandwidthGBPerSecond: 1008,
    ...overrides,
  });
}

test('converts decimal GB/s to GiB/s', () => {
  approx(decimalGBToGiB(1008), 1008 * 1e9 / (1024 ** 3));
});

test('calculates a fully resident model configuration', () => {
  const result = profile();

  assert.equal(result.fits, true);
  assert.equal(result.offloadFraction, 0);
  assert.equal(result.cpuOffloadPossible, true);
  approx(result.weightsRaw, expectedWeightsRaw);
  approx(result.weights, expectedWeights);
  approx(result.kvPerTokenGB, expectedKvPerTokenGB);
  approx(result.kv, expectedKv);
  approx(result.total, expectedTotal);
  assert.ok(result.maxContext > 0);
  approx(result.maxContext, expectedMaxContext);
  approx(result.tps, result.idealTps);
});

test('calculates viable partial CPU weight offload with a harmonic bandwidth mean', () => {
  const capacityGB = expectedKv + 1.5 + expectedWeights * 0.5;
  const result = profile({ capacityGB, systemRamBandwidthGBPerSecond: 60 });
  const expectedBandwidth = 1 / (0.5 / 1008 + 0.5 / 60);

  assert.equal(result.fits, false);
  assert.equal(result.cpuOffloadPossible, true);
  approx(result.offloadFraction, 0.5);
  approx(result.tps, (expectedBandwidth * 1e9 / (1024 ** 3)) / expectedWeights);
});

test('rejects CPU offload when KV cache plus runtime overhead exceed capacity', () => {
  const result = profile({ context: 131072, batchSize: 16, capacityGB: 1 });

  assert.equal(result.cpuOffloadPossible, false);
  assert.ok(result.nonWeightMemory > 1);
  assert.equal(result.tps, null);
  assert.equal(result.aggregateTps, null);
});

test('counts exact capacity equality as fitting', () => {
  const first = profile();
  const result = profile({ capacityGB: first.total });

  assert.equal(result.total, first.total);
  assert.equal(result.fits, true);
  approx(result.remaining, 0);
});

test('batch size increases modeled aggregate throughput with the existing exponent', () => {
  const single = profile({ batchSize: 1 });
  const batched = profile({ batchSize: 4 });

  approx(batched.tps, single.tps);
  approx(batched.aggregateTps, batched.tps * Math.pow(4, 0.85));
  assert.ok(batched.aggregateTps > single.aggregateTps);
});

test('disabling Flash Attention adds 25% of KV cache as scratch overhead', () => {
  const enabled = profile({ flashAttention: true });
  const disabled = profile({ flashAttention: false });

  approx(disabled.kv, enabled.kv);
  approx(disabled.attentionScratch, disabled.kv * 0.25);
  approx(disabled.overhead, 1.5 + disabled.kv * 0.25);
});

test('bounds offload fraction between zero and one', () => {
  const resident = profile();
  const beyondFullyOffloaded = profile({ capacityGB: 0.5 });
  const rawOffloadFraction = (expectedTotal - 0.5) / expectedWeights;

  assert.equal(resident.offloadFraction, 0);
  assert.ok(rawOffloadFraction > 1);
  assert.equal(beyondFullyOffloaded.offloadFraction, 1);
});

test('maximum context remains non-negative', () => {
  const result = profile({ capacityGB: 0.5 });

  assert.equal(result.maxContext, 0);
});
