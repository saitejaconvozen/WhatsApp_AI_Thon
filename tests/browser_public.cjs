const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(require.resolve('playwright', { paths: [process.cwd(), path.resolve(path.dirname(process.execPath), '..')] }));

async function main() {
  const url = process.argv[2] || 'http://127.0.0.1:8769';
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto(url);
    await page.locator('#deepseek .pilot-metrics').waitFor();
    await page.locator('#training .pilot-metrics').first().waitFor();
    const response = await page.request.get(new URL('/results.json', url).href);
    assert(response.ok());
    const report = await response.json();
    assert(report.published);
    assert.deepEqual(report.errors.items, []);
    function inspect(value) {
      if (!value || typeof value !== 'object') return;
      for (const [key, child] of Object.entries(value)) {
        assert(!['body', 'header', 'footer', 'buttons', 'sample_ids', 'predictions', 'base_url', 'api_key'].includes(key), `Private field: ${key}`);
        inspect(child);
      }
    }
    inspect(report);
    for (const route of ['/api/records', '/.env', '/.data/templates.sqlite3']) {
      assert.equal((await page.request.get(new URL(route, url).href)).status(), 404, route);
    }
    fs.mkdirSync('artifacts', { recursive: true });
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 900 });
      await page.locator('#training').screenshot({ path: `artifacts/public-training-${width}.png` });
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Horizontal overflow');
    }
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ url, status: 'passed', private_routes: '404', browser_errors: errors }));
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
