import test from 'node:test';
import assert from 'node:assert/strict';

import { profileInputsFromAssumptions, resolveAssumptions } from '../src/assumptions.js';
import { calculateProfile } from '../src/calculations.js';
import { BACKENDS } from '../src/data/backends.js';

const arch = {
  params: 8,
  layers: 32,
  kvHeads: 8,
  headDim: 128,
};

function inputs(overrides = {}) {
  return {
    arch,
    bitsPerWeight: 4.85,
    kvBytesPerElement: 2,
    context: 8192,
    batchSize: 1,
    flashAttention: true,
    capacityGB: 24,
    bandwidthGBPerSecond: 1008,
    ...overrides,
  };
}

test('the default assumption bag reproduces calculateProfile defaults', () => {
  const bag = resolveAssumptions();

  assert.equal(bag.systemRamBandwidthGBPerSecond, 60);
  assert.equal(bag.baselineOverheadGB, 1.5);
  assert.equal(bag.attentionScratchFactor, 0.25);
  assert.equal(bag.cpuOffloadMode, 'weights-if-kv-fits');
  assert.equal(bag.memory.id, 'generic-ddr');
  assert.equal(bag.memory.mode, 'spill');
  assert.equal(bag.backend.id, 'generic');
  assert.equal(
    bag.marginBuckets.reduce((total, bucket) => total + bucket.gb, 0),
    bag.baselineOverheadGB,
  );

  assert.deepEqual(
    calculateProfile({ ...inputs(), ...profileInputsFromAssumptions(bag) }),
    calculateProfile(inputs()),
  );
});

test('a custom memory profile changes only the spill ceiling', () => {
  const custom = resolveAssumptions({
    memoryId: 'custom',
    systemBandwidthGBPerSecond: 180,
  });
  const residentCustom = calculateProfile({
    ...inputs(),
    ...profileInputsFromAssumptions(custom),
  });
  const residentDefault = calculateProfile(inputs());

  assert.equal(residentCustom.tps, residentDefault.tps);
  assert.equal(residentCustom.total, residentDefault.total);

  const capacityGB = residentDefault.kv + residentDefault.baselineOverheadGB + residentDefault.weights * 0.5;
  const spillCustom = calculateProfile({
    ...inputs({ capacityGB }),
    ...profileInputsFromAssumptions(custom),
  });
  const spillDefault = calculateProfile({
    ...inputs({ capacityGB }),
    ...profileInputsFromAssumptions(resolveAssumptions()),
  });

  assert.equal(spillCustom.total, spillDefault.total);
  assert.equal(spillCustom.weights, spillDefault.weights);
  assert.ok(spillCustom.tps > spillDefault.tps);
});

test('invalid memory selections are rejected before they reach the engine', () => {
  assert.throws(
    () => resolveAssumptions({ memoryId: 'custom', systemBandwidthGBPerSecond: 0 }),
    RangeError,
  );
  assert.throws(
    () => resolveAssumptions({ memoryId: 'custom', systemBandwidthGBPerSecond: Number.NaN }),
    RangeError,
  );
  assert.throws(() => resolveAssumptions({ memoryId: 'ddr5-desktop' }), RangeError);
  assert.throws(() => resolveAssumptions({ memoryId: 'ddr5-server' }), RangeError);
});

test('backends do not change overhead or system bandwidth', () => {
  const baseline = resolveAssumptions();

  for (const backend of BACKENDS) {
    const bag = resolveAssumptions({ backendId: backend.id });
    assert.equal(bag.baselineOverheadGB, baseline.baselineOverheadGB);
    assert.equal(bag.systemRamBandwidthGBPerSecond, baseline.systemRamBandwidthGBPerSecond);
    assert.equal(bag.attentionScratchFactor, baseline.attentionScratchFactor);
  }

  assert.equal(resolveAssumptions({ backendId: 'llama.cpp' }).cpuOffloadMode, 'weights-if-kv-fits');
  assert.equal(resolveAssumptions({ backendId: 'ollama' }).cpuOffloadMode, 'weights-if-kv-fits');
  assert.equal(resolveAssumptions({ backendId: 'mlx' }).cpuOffloadMode, 'weights-if-kv-fits');
  assert.equal(resolveAssumptions({ backendId: 'exllamav2' }).cpuOffloadMode, 'disabled');
  assert.equal(resolveAssumptions({ backendId: 'vllm' }).cpuOffloadMode, 'disabled');
});

test('ExLlamaV2 and vLLM withhold the offload ceiling without scaling device bandwidth', () => {
  const resident = calculateProfile(inputs());
  const spillingCapacity = resident.kv + resident.baselineOverheadGB + resident.weights * 0.5;

  for (const backendId of ['exllamav2', 'vllm']) {
    const bag = resolveAssumptions({ backendId });
    const residentBackend = calculateProfile({
      ...inputs(),
      ...profileInputsFromAssumptions(bag),
    });
    const spilledBackend = calculateProfile({
      ...inputs({ capacityGB: spillingCapacity }),
      ...profileInputsFromAssumptions(bag),
    });
    const spilledGeneric = calculateProfile(inputs({ capacityGB: spillingCapacity }));

    assert.equal(residentBackend.tps, resident.tps);
    assert.equal(residentBackend.total, resident.total);
    assert.equal(residentBackend.overhead, resident.overhead);
    assert.equal(spilledBackend.tps, null);
    assert.equal(spilledBackend.total, spilledGeneric.total);
    assert.ok(spilledGeneric.tps > 0);
  }
});
