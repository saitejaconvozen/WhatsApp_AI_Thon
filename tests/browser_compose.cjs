const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || require.resolve('playwright', { paths: [process.cwd(), path.resolve(path.dirname(process.execPath), '..')] }));

async function main() {
  fs.mkdirSync('artifacts', { recursive: true });
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const page = await browser.newPage({ baseURL: process.env.TEMPLATE_LAB_URL || 'http://127.0.0.1:8767' });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    const llm = await page.request.get('/api/llm').then(r => r.json());
    assert(!llm.available, 'Browser tests must not issue paid provider calls');
    await page.goto('/#compose');
    await page.locator('#compose-body').fill('Your order {{1}} has shipped. 20% off your next order!');
    await page.locator('#compose-check').click();
    await page.getByText('Needs more context', { exact: true }).waitFor();
    await page.getByRole('heading', { name: 'Predicted Meta category' }).waitFor();
    await page.locator('#compose-relationship').check();
    await page.locator('#compose-check').click();
    await page.locator('#compose-accept').waitFor();
    await page.locator('#compose-accept').click();
    assert.equal(await page.locator('#compose-body').inputValue(), 'Your order {{1}} has shipped.');
    await page.locator('[data-mode="generate"]').click();
    assert.equal(await page.locator('[data-mode="generate"]').getAttribute('aria-pressed'), 'true');
    await page.locator('#compose-task').fill('Tell the customer their invoice is ready');
    await page.locator('#compose-context').fill('Existing customer and an issued invoice.');
    await page.locator('#compose-generate').click();
    await page.locator('#compose-send-to-check').waitFor();
    await page.locator('#toast').waitFor({ state: 'hidden' });
    assert(await page.locator('.template-card').innerText().then(t => t.includes('{{')));
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 900 });
      await page.screenshot({ path: `artifacts/compose-pilot-${width}.png`, fullPage: true });
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Compose overflow');
    }
    await page.locator('#compose-task').fill('Offer 20% off the next order');
    await page.locator('#compose-context').fill('');
    await page.locator('#compose-generate').click();
    await page.getByText('Cannot become utility', { exact: true }).waitFor();
    await page.goto('/#benchmark');
    await page.getByText('73.0%', { exact: true }).waitFor();
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 900 });
      await page.screenshot({ path: `artifacts/deepseek-pilot-${width}.png`, fullPage: true });
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Pilot overflow');
    }
    await page.goto(process.env.RESULTS_URL || 'http://127.0.0.1:8766');
    await page.locator('#deepseek').waitFor();
    assert((await page.locator('#deepseek').innerText()).includes('73.0%'));
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 900 });
      await page.locator('#deepseek').screenshot({ path: `artifacts/results-pilot-${width}.png` });
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Results overflow');
    }
    assert.deepEqual(errors, []);
    console.log('PASS: classification, human context, approved edit, intent/context drafting, promotion refusal, and pilot reports on desktop/mobile.');
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
