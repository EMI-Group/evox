// Capture the website's original evolution diagram, without redrawing its layout.
// Build the pinned portal checkout first. Requires its installed Playwright and Python.
// node docs/tools/export-readme-evolution.mjs /path/to/evox-portal /path/to/frames
// python docs/tools/encode_readme_media.py /path/to/frames
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { execFileSync, spawn } from 'node:child_process';

const portal = path.resolve(process.argv[2]);
const output = path.resolve(process.argv[3]);
const assets = path.resolve(import.meta.dirname, '../source/_static');
const previous = JSON.parse(readFileSync(path.join(assets, 'readme-motion-source.json'), 'utf8'));
const revision = previous.revision;
const source = file => execFileSync('git', ['show', `${revision}:${file}`], { cwd: portal, encoding: 'utf8' });
const script = source('src/components/EvolutionDiagram.astro').match(/<script is:inline>([\s\S]*?)<\/script>/)[1];
const { chromium } = await import(pathToFileURL(path.join(portal, 'node_modules/playwright/index.mjs')));
const browser = await chromium.launch({ headless: true });
const server = spawn('python', ['-m', 'http.server', '4331', '--bind', '127.0.0.1', '--directory', path.join(portal, 'dist')], { stdio: 'ignore', windowsHide: true });
const frameMs = 20, pixelRatio = 2, displaySize = 400;
const manifest = {
  ...previous,
  sourceFiles: [...new Set([...previous.sourceFiles, 'src/components/EvolutionDiagram.astro'])],
  evolutionAdaptation: 'Direct capture of the complete original website component: square diagram, three step labels and short caption beneath. Controls are hidden and caption height is reserved for two lines. Manual 50 fps clock, 400 px display width and 2x pixel density. Two original generations and a 400 ms replay crossfade; the static view shows inheritance.',
  scenes: [],
};
mkdirSync(output, { recursive: true });
try {
  await new Promise(resolve => setTimeout(resolve, 600));
  for (const locale of ['en', 'zh-cn']) {
    const page = await browser.newPage({ viewport: { width: 960, height: 800 }, deviceScaleFactor: pixelRatio });
    await page.goto(`http://127.0.0.1:4331/${locale === 'en' ? '' : locale}`, { waitUntil: 'networkidle' });
    await page.evaluate(() => {
      window.evoxLoop.destroy();
      const root = document.querySelector('.evox-evolution').cloneNode(true);
      document.body.replaceChildren(root);
      document.documentElement.dataset.theme = 'light';
    });
    await page.addStyleTag({ content: `body{margin:0!important;padding:0!important}.evox-evolution{width:${displaySize}px!important;max-width:none!important}.evx-controls{display:none!important}.evx-caption{min-height:3.2em!important}` });
    await page.addScriptTag({ content: script });
    await page.evaluate(() => {
      const loop = window.evoxLoop;
      loop.pause(); loop.resize();
      loop.resizeObserver.disconnect(); loop.intersectionObserver.disconnect();
      const parents = loop.generation(0), children = loop.generation(1);
      const retained = children.items.filter(item => item.retained);
      if (retained.length !== 3 || !retained.every(item => parents.winners.some(i => parents.items[i].id === item.id))) throw new Error('Selected parents were lost');
      if (children.items.filter(item => !item.retained).length !== 9) throw new Error('Expected nine descendants');
      window.readmeSeek = ms => { loop.time = ms; loop.playing = true; loop.render(); loop.playing = false; };
    });
    await page.evaluate(() => document.fonts.ready);
    const cycle = await page.evaluate(() => window.evoxLoop.getState().cycleMilliseconds);
    const duration = cycle * 2 + 400, frames = duration / frameMs;
    const name = `readme-evolution-loop-${locale === 'en' ? 'en' : 'zh'}`;
    const folder = path.join(output, name); mkdirSync(folder, { recursive: true });
    const visual = page.locator('.evox-evolution');
    await page.evaluate(ms => window.readmeSeek(ms), cycle + 1800);
    await visual.screenshot({ path: path.join(assets, name + '.png') });
    for (let index = 0; index < cycle * 2 / frameMs; index++) {
      await page.evaluate(ms => window.readmeSeek(ms), index * frameMs);
      await visual.screenshot({ path: path.join(folder, String(index).padStart(4, '0') + '.png') });
    }
    // Blend only the replay boundary; every other frame is the actual website visual.
    const first = readFileSync(path.join(folder, '0000.png')).toString('base64');
    const last = readFileSync(path.join(folder, String(cycle * 2 / frameMs - 1).padStart(4, '0') + '.png')).toString('base64');
    const replay = await page.evaluate(async ({ first, last, frameMs }) => {
      const images = await Promise.all([first, last].map(data => new Promise(resolve => { const img = new Image(); img.onload = () => resolve(img); img.src = 'data:image/png;base64,' + data; })));
      const canvas = document.createElement('canvas'); canvas.width = images[0].naturalWidth; canvas.height = images[0].naturalHeight;
      const ctx = canvas.getContext('2d');
      return Array.from({ length: 400 / frameMs }, (_, i) => {
        ctx.globalAlpha = 1; ctx.drawImage(images[1], 0, 0);
        const t = (i + 1) * frameMs / 400; ctx.globalAlpha = t * t * (3 - 2 * t); ctx.drawImage(images[0], 0, 0);
        return canvas.toDataURL('image/png');
      });
    }, { first, last, frameMs });
    replay.forEach((frame, i) => writeFileSync(path.join(folder, String(cycle * 2 / frameMs + i).padStart(4, '0') + '.png'), Buffer.from(frame.split(',')[1], 'base64')));
    const png = readFileSync(path.join(folder, '0000.png'));
    manifest.scenes.push({ name, kind: 'evolution', locale, frames, durationMs: duration, phaseDurationsMs: [5000, 3400, 3000], width: png.readUInt32BE(16), height: png.readUInt32BE(20), displayWidth: displaySize, displayHeight: png.readUInt32BE(20) / pixelRatio, pixelRatio, sourceRevision: revision });
    console.log(`${name}: ${frames} website frames, ${duration} ms`);
    await page.close();
  }
  writeFileSync(path.join(output, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
} finally {
  await browser.close(); server.kill();
}
