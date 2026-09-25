/* ------------------------------------------------------------------ *
 * System-memory profiles for weights that spill out of device memory.
 * Bandwidth numbers are decimal GB/s. Only profiles with a source, or
 * an explicit custom value, are selectable. Apple Silicon and GB10 keep
 * the generic spill figure until a unified-pool profile is chosen;
 * that profile is not part of this catalog.
 * ------------------------------------------------------------------ */
export const DEFAULT_MEMORY_ID = 'generic-ddr';

export const MEMORY_PROFILES = [
  {
    id: 'generic-ddr',
    label: 'Generic DDR',
    mode: 'spill',
    bandwidthGBPerSecond: 60,
    selectable: true,
    note: 'Default spill bandwidth of 60 GB/s for every preset, including Apple Silicon and GB10.',
  },
  {
    id: 'custom',
    label: 'Custom',
    mode: 'spill',
    selectable: true,
    note: 'Same spill math as generic DDR, using a positive finite GB/s you enter.',
  },
  {
    id: 'ddr5-desktop',
    label: 'DDR5 desktop',
    mode: 'spill',
    selectable: false,
    note: 'Reserved name. No bandwidth is stored until a source is written next to this profile.',
  },
  {
    id: 'ddr5-server',
    label: 'DDR5 server',
    mode: 'spill',
    selectable: false,
    note: 'Reserved name. No bandwidth is stored until a source is written next to this profile.',
  },
];

export function selectableMemoryProfiles() {
  return MEMORY_PROFILES.filter((profile) => profile.selectable);
}
