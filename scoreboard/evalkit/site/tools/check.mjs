// Local launch-check pass for the scorecard site (item D9). No em dashes.
// Needs playwright@1.49.1 and axe-core@4.10.2 installed in the folder you run from, and the site served by
//   python -m scoreboard.evalkit.site.serve <site> 8795
// Run: node <kit>/scoreboard/evalkit/site/tools/check.mjs http://127.0.0.1:8795 <out.json>
// Checks every sitemap page plus the 404 page at 1440x900 and 390x844: axe (wcag2a, wcag2aa, wcag21aa,
// best-practice), console errors and warnings with the real CSP on, the page with JavaScript off, horizontal
// overflow at phone width, and the status of every link target on the site (internal) and outside (HEAD/GET).
import { createRequire } from 'node:module';
import { readFileSync, writeFileSync } from 'node:fs';
const req = createRequire(process.cwd() + '/');
const { chromium, request } = req('playwright');
const axeSrc = readFileSync(req.resolve('axe-core/axe.min.js'), 'utf8');
const [,, base, out] = process.argv;

const sm = await (await fetch(base + '/sitemap.xml')).text();
const paths = [...sm.matchAll(/<loc>https:\/\/scorecard\.geoliquefy\.com(\/[^<]*)<\/loc>/g)].map((m) => m[1]);
paths.push('/no-such-page/');
const sizes = [{ width: 1440, height: 900 }, { width: 390, height: 844 }];
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const result = { pages: paths.length, axe: [], console: [], nojs: [], overflow: [], status: {}, links: {} };
const links = new Set();
for (const vp of sizes) {
  const strict = await browser.newContext({ viewport: vp });
  const loose = await browser.newContext({ viewport: vp, bypassCSP: true });
  for (const p of paths) {
    const page = await strict.newPage();
    page.on('console', (m) => { if (['error', 'warning'].includes(m.type())) result.console.push({ p, vp: vp.width, type: m.type(), text: m.text() }); });
    const r = await page.goto(base + p, { waitUntil: 'networkidle' });
    result.status[p] = r.status();
    const ov = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    if (ov > 0) result.overflow.push({ p, vp: vp.width, px: ov });
    if (p !== '/no-such-page/') for (const h of await page.$$eval('a[href], link[href]:not([rel=preconnect]), img[src]', (es) => es.map((e) => e.href || e.src))) links.add(h);
    await page.close();
    const a = await loose.newPage();
    await a.goto(base + p, { waitUntil: 'networkidle' });
    await a.addScriptTag({ content: axeSrc });
    const v = await a.evaluate(async () => (await window.axe.run(document, { runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21aa', 'best-practice'] } })).violations.map((x) => ({ id: x.id, impact: x.impact, n: x.nodes.length, target: x.nodes[0] && x.nodes[0].target })));
    for (const x of v) result.axe.push({ p, vp: vp.width, ...x });
    await a.close();
  }
  await strict.close(); await loose.close();
}
const nojs = await browser.newContext({ javaScriptEnabled: false });
for (const p of paths) {
  const page = await nojs.newPage();
  await page.goto(base + p);
  const h1 = await page.$eval('h1', (e) => e.textContent.trim()).catch(() => '');
  if (!h1) result.nojs.push(p);
  await page.close();
}
await browser.close();
const api = await request.newContext();
for (const u of [...links].sort()) {
  if (!u.startsWith('http')) continue;
  const uu = u.replace('https://scorecard.geoliquefy.com', base);  // the site's own canonical URLs, checked locally
  try {
    let r = await api.head(uu, { maxRedirects: 5, timeout: 20000 });
    if (r.status() >= 400) r = await api.get(uu, { maxRedirects: 5, timeout: 20000 });
    result.links[u] = r.status();
  } catch (e) { result.links[u] = String(e).slice(0, 80); }
}
result.summary = {
  axe_violations: result.axe.length, console: result.console.length, nojs_blank: result.nojs.length,
  overflow: result.overflow.length,
  bad_status: Object.entries(result.status).filter(([p, s]) => (p === '/no-such-page/' ? s !== 404 : s !== 200)),
  bad_links: Object.entries(result.links).filter(([, s]) => s !== 200),
};
writeFileSync(out, JSON.stringify(result, null, 1));
console.log(JSON.stringify(result.summary, null, 1));
