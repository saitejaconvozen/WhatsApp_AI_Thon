const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(require.resolve('playwright', { paths: [process.cwd(), path.resolve(path.dirname(process.execPath), '..')] }));

async function main() {
  const url = process.argv[2] || 'http://127.0.0.1:8770';
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto(url);
    await page.getByLabel('Category submitted to Meta').selectOption('UTILITY');
    await page.getByLabel('Body', { exact: true }).fill('Your invoice {{invoice_id}} is due on {{due_date}}.');
    await page.getByRole('button', { name: 'Classify template' }).click();
    await page.locator('#result .category').waitFor();
    assert(['UTILITY', 'MARKETING'].includes(await page.locator('#result .category').innerText()));
    assert((await page.locator('#result').innerText()).includes('Utility probability'));
    assert((await page.locator('#result').innerText()).includes('Supervised encoder'));
    fs.mkdirSync('artifacts', { recursive: true });
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 900 });
      await page.screenshot({ path: `artifacts/public-model-${width}.png`, fullPage: true });
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Horizontal overflow');
    }
    for (const route of ['/api/records', '/api/export', '/.env', '/.data/templates.sqlite3']) {
      assert.equal((await page.request.get(new URL(route, url).href)).status(), 404);
    }
    await page.getByRole('button', { name: 'Clear template' }).click();
    assert.equal(await page.getByLabel('Body', { exact: true }).inputValue(), '');
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ url, status: 'passed', prediction: 'rendered', private_routes: '404', browser_errors: errors }));
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
