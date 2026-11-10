// Node 24+, built portal checkout with Playwright, Python, then encode_readme_media.py.
// Finally run reframe_readme_evolution.py to start on a complete population.
// node docs/tools/export-current-readme-media.mjs PORTAL_CHECKOUT FRAME_DIRECTORY [scenes|evolution|compute]
// Uses the portal renderers unchanged; only playback clocks and export framing differ.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { stripTypeScriptTypes } from 'node:module';
import { execFileSync, spawn } from 'node:child_process';

const portal = path.resolve(process.argv[2]), output = path.resolve(process.argv[3]);
const only = process.argv[4], assets = path.resolve(import.meta.dirname, '../source/_static');
const revision = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: portal, encoding: 'utf8' }).trim();
const source = file => execFileSync('git', ['show', `${revision}:${file}`], { cwd: portal, encoding: 'utf8' });
const cleanTS = code => stripTypeScriptTypes(code.replace(/^\s*import .*$/gm, '')).replace(/^export /gm, '');
const helpers = ['src/components/art/animation-lifetime.ts', 'src/components/art/diagram-theme.ts'];
const sceneFiles = [...helpers, 'src/components/art/recursive-agents.ts', 'src/components/art/paradigm-scenes.ts'];
const sourceFiles = [...sceneFiles, 'src/components/art/ParadigmScene.astro', 'src/components/EvolutionDiagram.astro', 'src/components/ParallelDiagram.astro', 'src/styles/global.css', 'src/i18n/locales/en.json', 'src/i18n/locales/zh-cn.json'];
const manifest = {
  repository: 'https://github.com/EMI-Group/evox-portal', revision, sourceFiles,
  theme: 'light', fps: 50, frameDurationMs: 20,
  objectives: 'Illustrative deterministic animations, not measured model, robotics or CFD results.',
  adaptation: 'Unchanged production renderers and original phase timings. Manual 50 fps clock; hidden playback controls. Application canvases have localized captions and a fixed export frame. Evolution and hardware use direct component screenshots. Two original generations/agent episodes, followed by a 400 ms replay crossfade; hardware runs one complete 13000 ms cycle.',
  evolutionAdaptation: 'Direct capture of the complete production evolution component at 400 px and 2x density, with step labels and caption. The static view shows inheritance.',
  scenes: [],
};
mkdirSync(output, { recursive: true });
const { chromium } = await import(pathToFileURL(path.join(portal, 'node_modules/playwright/index.mjs')));
const browser = await chromium.launch({ headless: true });
// The portal remembers browser language; keep exports explicitly English or Chinese.
const server = spawn('python', ['-m', 'http.server', '4336', '--bind', '127.0.0.1', '--directory', path.join(portal, 'dist')], { stdio: 'ignore', windowsHide: true });
async function record(page, name, kind, locale, cycle, stillAt, visual) {
  const folder = path.join(output, name); mkdirSync(folder, { recursive: true });
  const contentMs = kind === 'compute' ? cycle : cycle * 2;
  const durationMs = contentMs + (kind === 'compute' ? 0 : 400);
  const getFrame = async ms => {
    if (visual) { await page.evaluate(ms => window.readmeSeek(ms), ms); return visual.screenshot(); }
    return Buffer.from((await page.evaluate(ms => window.readmeFrame(ms), ms)).split(',')[1], 'base64');
  };
  writeFileSync(path.join(assets, name + '.png'), await getFrame(stillAt));
  if (!visual) await page.evaluate(() => window.readmeReset());
  let first, last;
  for (let i = 0; i < contentMs / 20; i++) {
    const frame = await getFrame(i * 20); first ??= frame; last = frame;
    writeFileSync(path.join(folder, String(i).padStart(4, '0') + '.png'), frame);
    if (i % 250 === 0) console.log(`${name}: frame ${i}/${contentMs / 20}`);
  }
  if (durationMs > contentMs) {
    const frames = await page.evaluate(async ({ first, last }) => {
      const images = await Promise.all([first, last].map(data => new Promise(resolve => { const img = new Image(); img.onload = () => resolve(img); img.src = 'data:image/png;base64,' + data; })));
      const canvas = document.createElement('canvas'); canvas.width = images[0].width; canvas.height = images[0].height;
      const ctx = canvas.getContext('2d');
      return Array.from({ length: 20 }, (_, i) => {
        ctx.globalAlpha = 1; ctx.drawImage(images[1], 0, 0);
        const t = (i + 1) / 20; ctx.globalAlpha = t * t * (3 - 2 * t); ctx.drawImage(images[0], 0, 0);
        return canvas.toDataURL('image/png');
      });
    }, { first: first.toString('base64'), last: last.toString('base64') });
    frames.forEach((data, i) => writeFileSync(path.join(folder, String(contentMs / 20 + i).padStart(4, '0') + '.png'), Buffer.from(data.split(',')[1], 'base64')));
  }
  manifest.scenes.push({ name, kind, locale, sourceRevision: revision, frames: durationMs / 20, durationMs, width: first.readUInt32BE(16), height: first.readUInt32BE(20) });
  console.log(`${name}: complete`);
}
try {
  await new Promise(resolve => setTimeout(resolve, 600));
  for (const locale of ['en', 'zh-cn']) {
    const url = `http://127.0.0.1:4336/${locale === 'en' ? '' : locale}`;
    if (!only || only === 'scenes') {
      const page = await browser.newPage({ viewport: { width: 1100, height: 850 } });
      await page.addInitScript(locale => localStorage.setItem('evox-locale', locale), locale);
      await page.goto(url, { waitUntil: 'networkidle' });
      await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
      await page.addScriptTag({ content: cleanTS(sceneFiles.map(source).join('\n')) + '\nwindow.ReadmeScene=ParadigmScene;window.readmeNext=nextPopulation;window.readmePalette=readDiagramPalette;' });
      for (const kind of ['generative', 'agentic', 'physical', 'science']) {
        await page.evaluate(kind => {
          const root = document.querySelector(`.paradigm-scene[data-kind="${kind}"]`).cloneNode(true);
          const canvas = document.createElement('canvas'); canvas.width = 720; canvas.height = 420;
          const frame = document.createElement('canvas'); frame.width = 720; frame.height = 484;
          const fc = frame.getContext('2d'), palette = window.readmePalette();
          const scene = Object.create(window.ReadmeScene.prototype);
          Object.assign(scene, { root, canvas, ctx: canvas.getContext('2d'), kind, caption: document.createElement('p'), palette, ratio: 1, scale: 1.5, durations: [2200,4200,1800,2200,4200], starts: [0,2200,6400,8200,10400,14600] });
          window.readmeReset = () => Object.assign(scene, { generation: 0, time: 0, lastCaption: -1, previousPopulation: undefined, offspring: undefined, population: kind === 'agentic' ? { candidates: [], winner: 0 } : window.readmeNext(kind, undefined, 0) });
          window.readmeReset();
          window.readmeFrame = ms => {
            const generation = Math.floor(ms / 14600);
            if (generation < scene.generation) window.readmeReset();
            while (scene.generation < generation) {
              scene.previousPopulation = scene.population; scene.generation++;
              if (kind !== 'agentic') scene.population = scene.offspring ?? window.readmeNext(kind, scene.population, scene.generation);
              scene.offspring = undefined;
            }
            scene.time = ms % 14600; scene.render();
            fc.fillStyle = palette.bg; fc.fillRect(0,0,720,484); fc.drawImage(canvas,0,0);
            fc.fillStyle = palette.muted; fc.font = '18px system-ui, sans-serif'; fc.textAlign = 'center';
            // Break by Unicode character as well as word boundaries for Chinese captions.
            const lines = []; let line = '';
            for (const char of scene.caption.textContent) { if (fc.measureText(line + char).width > 672) { lines.push(line.trim()); line = ''; } line += char; }
            lines.push(line.trim()); lines.forEach((value,i) => fc.fillText(value,360,443+i*24));
            fc.strokeStyle = palette.line; fc.beginPath(); fc.roundRect(.5,.5,719,483,18); fc.stroke();
            return frame.toDataURL('image/png');
          };
        }, kind);
        await record(page, `readme-${kind}${locale === 'en' ? '' : '-zh'}`, kind, locale, 14600, kind === 'agentic' ? 12000 : 13700);
      }
      await page.close();
    }
    if (!only || only === 'evolution') {
      const page = await browser.newPage({ viewport: { width: 960, height: 850 }, deviceScaleFactor: 2 });
      await page.addInitScript(locale => localStorage.setItem('evox-locale', locale), locale);
      await page.goto(url, { waitUntil: 'networkidle' });
      await page.evaluate(() => {
        window.evolutionDiagram.destroy();
        document.body.replaceChildren(document.querySelector('.evox-evolution').cloneNode(true));
        document.documentElement.dataset.theme = 'light';
      });
      await page.addStyleTag({ content: 'body{margin:0!important;padding:0!important}.evox-evolution{width:400px!important;max-width:none!important}.evx-controls{display:none!important}.evx-caption{min-height:3.2em!important}' });
      await page.addScriptTag({ content: source('src/components/EvolutionDiagram.astro').match(/<script is:inline>([\s\S]*?)<\/script>/)[1] });
      await page.evaluate(() => {
        const loop = window.evolutionDiagram; loop.pause(); loop.resize(); loop.resizeObserver.disconnect(); loop.intersectionObserver.disconnect();
        window.readmeSeek = ms => { loop.time = ms; loop.playing = true; loop.render(); loop.playing = false; };
      });
      await page.evaluate(() => document.fonts.ready);
      await record(page, `readme-evolution-loop-${locale === 'en' ? 'en' : 'zh'}`, 'evolution', locale, 11400, 3600, page.locator('.evox-evolution'));
      await page.close();
    }
    if (!only || only === 'compute') {
      const page = await browser.newPage({ viewport: { width: 1100, height: 850 } });
      await page.addInitScript(locale => localStorage.setItem('evox-locale', locale), locale);
      await page.goto(url, { waitUntil: 'networkidle' });
      await page.evaluate(() => {
        document.documentElement.dataset.theme = 'light';
        const figure = document.querySelector('.parallel-figure');
        if (!figure.hasAttribute('data-user-paused')) figure.querySelector('.motion-toggle').click();
        figure.replaceWith(figure.cloneNode(true));
      });
      await page.addStyleTag({ content: '.parallel-figure{width:960px!important;margin:0!important}.motion-controls{display:none!important}' });
      const body = source('src/components/ParallelDiagram.astro').match(/<script>([\s\S]*?)<\/script>/)[1]
        .replace('if (!paused) frame = requestAnimationFrame(tick);', '')
        .replace('measure(); sync();', 'measure(); sync(); window.readmeSeek = ms => { elapsed = ms; draw(); };');
      await page.addScriptTag({ content: cleanTS(helpers.map(source).join('\n') + '\n' + body) });
      const visual = page.locator('.parallel-figure'); await visual.scrollIntoViewIfNeeded(); await page.evaluate(() => document.fonts.ready);
      await record(page, `readme-compute-${locale === 'en' ? 'en' : 'zh'}`, 'compute', locale, 13000, 11200, visual);
      await page.close();
    }
  }
  writeFileSync(path.join(output, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
} finally { await browser.close(); server.kill(); }
