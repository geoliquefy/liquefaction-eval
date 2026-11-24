// Renders og.png, the 1200x630 link-preview image, from the real front page (header, title, first list).
// No em dashes. Run where playwright@1.49.1 is installed, with the site served by
//   python -m scoreboard.evalkit.site.serve <site> 8795
//   node <kit>/scoreboard/evalkit/site/tools/og.mjs http://127.0.0.1:8795 /tmp/og-2x.png
//   sips -s format png -z 630 1200 /tmp/og-2x.png --out <kit>/scoreboard/evalkit/site/static/og.png
import { createRequire } from 'node:module';
const { chromium } = createRequire(process.cwd() + '/')('playwright');
const [,, base, out] = process.argv;
const b = await chromium.launch({ channel: 'chrome', headless: true });
const p = await b.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 2, colorScheme: 'light' });
await p.goto(base + '/', { waitUntil: 'networkidle' });
await p.evaluate(() => document.fonts.ready);
await p.screenshot({ path: out });
await b.close();
