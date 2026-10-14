/** Render paper figures first with pdftoppm; this script trims page whitespace. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const sharp = require('sharp');
const root = path.resolve(import.meta.dirname, '..');
const [renders, sysu, hkust] = process.argv.slice(2);
if (!renders || !sysu || !hkust) {
  throw new Error('Usage: node scripts/prepare_project_media.mjs RENDER_DIRECTORY SYSU_PNG HKUST_PNG');
}
const destination = path.join(root, 'docs', 'assets');
await fs.mkdir(destination, { recursive: true });
const figures = {
  fig_teaser: 'overview', agentic_rsr: 'metric-alignment', sim_real: 'real-sim',
  real_robot_execution: 'robot-execution', policy_iteration_loop: 'policy-loop',
};
for (const [source, target] of Object.entries(figures)) {
  const input = path.join(renders, `${source}.png`);
  await sharp(input).trim({ threshold: 12 }).extend({ top: 12, bottom: 12, left: 12, right: 12,
    background: '#ffffff' }).png().toFile(path.join(destination, `${target}.png`));
}
// Official header marks are white on transparent backgrounds; recolor for white pages.
for (const [source, target, color] of [
  [sysu, 'sysu-logo.png', [0, 91, 51]],
  [hkust, 'hkust-gz-logo.png', [0, 57, 88]],
]) {
  const { data, info } = await sharp(source).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  for (let index = 0; index < data.length; index += 4) {
    data[index] = color[0]; data[index + 1] = color[1]; data[index + 2] = color[2];
  }
  await sharp(data, { raw: info }).flatten({ background: '#ffffff' }).png().toFile(path.join(destination, target));
}
console.log(`Prepared ${Object.keys(figures).length} paper figures and two affiliation marks.`);
