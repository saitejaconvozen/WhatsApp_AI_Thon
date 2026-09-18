const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const playwrightPath = process.env.PLAYWRIGHT_MODULE || require.resolve('playwright', { paths: [process.cwd(), path.resolve(path.dirname(process.execPath), '..')] });
const { chromium } = require(playwrightPath);

async function main() {
  const output = path.resolve('artifacts');
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true, channel: process.env.PLAYWRIGHT_BROWSER_CHANNEL || 'chrome' });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, baseURL: process.env.TEMPLATE_LAB_URL || 'http://127.0.0.1:8765' });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto(process.env.TEMPLATE_LAB_URL || 'http://127.0.0.1:8765');
    await page.getByRole('heading', { name: 'Template Lab development', exact: true }).waitFor();
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
      await page.setViewportSize(viewport);
      for (const name of ['Progress', 'Data journey', 'How it works', 'Results', 'Code map']) {
        await page.getByRole('tab', { name, exact: true }).click();
        assert.equal(await page.getByRole('tab', { name, exact: true }).getAttribute('aria-selected'), 'true');
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), `Development overflow: ${name}`);
        await page.screenshot({ path: path.join(output, `development-${name.replaceAll(' ', '-')}-${viewport.width}.png`), fullPage: true });
      }
    }
    await page.getByRole('tab', { name: 'Progress', exact: true }).click();
    await page.keyboard.press('ArrowRight');
    assert.equal(await page.getByRole('tab', { name: 'Data journey', exact: true }).getAttribute('aria-selected'), 'true');
    await page.getByRole('button', { name: 'Refresh development data' }).click();
    await page.waitForFunction(() => !document.querySelector('#refresh-development').disabled);
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.locator('[data-nav="dataset"]').click();
    await page.getByRole('heading', { name: 'Template dataset', exact: true }).waitFor();
    await page.locator('.data-table tbody tr').first().waitFor();
    const summary = await page.request.get('/api/summary').then(response => response.json());
    assert(summary.total > 0);
    await page.screenshot({ path: path.join(output, 'dataset-desktop.png') });

    await page.getByLabel('Search templates').fill('this-message-is-not-in-the-dataset-zzyy');
    await page.getByRole('heading', { name: 'No matching templates' }).waitFor();
    await page.getByRole('button', { name: 'Clear filters' }).click();
    await page.locator('.data-table tbody tr').first().waitFor();
    await page.getByLabel('Filter by recorded category').selectOption('UTILITY');
    await page.waitForFunction(() => [...document.querySelectorAll('.data-table tbody tr')].every(row => row.children[2].textContent.trim() === 'Utility'));
    await page.getByRole('button', { name: 'Next page', exact: true }).click();
    await page.getByText(/Page 2 of/).waitFor();
    await page.getByRole('button', { name: 'Clear filters' }).click();
    await page.locator('button.template-name').first().click();
    await page.getByRole('button', { name: 'Review template', exact: true }).waitFor();
    await page.getByRole('button', { name: 'Close details', exact: true }).click();

    if (process.argv[2]) {
      await page.getByRole('button', { name: 'Import templates', exact: true }).click();
      await page.locator('#upload-file').setInputFiles(process.argv[2]);
      await page.getByRole('button', { name: /Import [\d,]+ records/ }).click();
      await page.waitForFunction(() => !document.querySelector('#import-dialog').open);
      const after = await page.request.get('/api/summary').then(response => response.json());
      assert.equal(after.total, summary.total, 'Reimport must not duplicate rows');
    }

    await page.getByRole('link', { name: 'Model lab', exact: true }).click();
    const model = await page.request.get('/api/model').then(response => response.json());
    if (!model.trained || model.stale) {
      await page.getByRole('button', { name: /^(Train|Retrain) baseline$/ }).click();
      await page.getByRole('heading', { name: 'Holdout confusion matrix' }).waitFor({ timeout: 120000 });
    }
    const report = await page.request.get('/api/model').then(response => response.json());
    assert(report.trained && !report.stale);
    assert.equal(report.report.group_overlap, 0);
    await page.screenshot({ path: path.join(output, 'model-desktop.png') });

    await page.locator('[data-nav="errors"]').click();
    await page.getByRole('heading', { name: 'Error review', exact: true }).waitFor();
    const errorsBefore = await page.request.get('/api/errors').then(response => response.json());
    const firstError = errorsBefore.items.find(row => row.direction === 'false_utility');
    assert(firstError);
    await page.locator(`[data-error-id="${firstError.id}"]`).click();
    try {
      await page.getByLabel('Review notes', { exact: true }).fill('Browser persistence check');
      await page.locator('[data-nav="experiments"]').click();
      await page.waitForFunction(() => location.hash === '#errors');
      assert.equal(await page.getByLabel('Review notes', { exact: true }).inputValue(), 'Browser persistence check');
      await page.getByRole('button', { name: 'Save review', exact: true }).click();
      await page.waitForFunction(() => document.querySelector('#review-save-state').textContent.startsWith('Saved version'));
      await page.reload();
      await page.getByLabel('Review notes', { exact: true }).waitFor();
      assert.equal(await page.getByLabel('Review notes', { exact: true }).inputValue(), 'Browser persistence check');
      await page.screenshot({ path: path.join(output, 'errors-desktop.png'), fullPage: true });
      await page.getByLabel('Review notes', { exact: true }).fill('Unsaved discard check');
      await page.getByRole('button', { name: 'Discard edits', exact: true }).click();
      assert.equal(await page.getByLabel('Review notes', { exact: true }).inputValue(), 'Browser persistence check');
      await page.getByText('Candidate draft and observed outcome', { exact: true }).click();
      await page.getByLabel('Candidate header', { exact: true }).fill('');
      await page.getByLabel('Candidate body', { exact: true }).fill('Your invoice {{id}} is due on {{date}}. Upgrade today.');
      await page.getByLabel('Candidate footer', { exact: true }).fill('');
      await page.getByLabel('Candidate buttons', { exact: true }).fill('View invoice');
      await page.getByLabel('Service event', { exact: true }).selectOption('billing');
      await page.locator('#annotation-relationship').check();
      await page.getByRole('button', { name: 'Assess candidate', exact: true }).click();
      await page.getByRole('button', { name: 'Use limited edit', exact: true }).click();
      assert.equal(await page.getByLabel('Candidate body', { exact: true }).inputValue(), 'Your invoice {{id}} is due on {{date}}.');
      assert.equal(await page.getByLabel('Observed Meta outcome', { exact: true }).inputValue(), 'not_submitted');
      await page.getByRole('button', { name: 'Discard edits', exact: true }).click();
    } finally {
      const current = await page.request.get('/api/errors').then(response => response.json());
      const annotation = current.items.find(row => row.id === firstError.id).annotation;
      const restored = await page.request.post(`/api/errors/${firstError.id}`, { data: {
        ...firstError.annotation, version: annotation.version, snapshot: current.snapshot,
      } });
      assert(restored.ok(), 'Restore original review after browser check');
    }
    await page.locator('[data-nav="experiments"]').click();
    await page.getByRole('heading', { name: 'Model experiments', exact: true }).waitFor();
    const experiments = await page.request.get('/api/experiments').then(response => response.json());
    if (!experiments.available || experiments.stale) {
      await page.getByRole('button', { name: 'Run comparison' }).click();
    }
    await page.getByRole('heading', { name: 'Full chronological holdout' }).waitFor({ timeout: 120000 });
    await page.screenshot({ path: path.join(output, 'experiments-desktop.png'), fullPage: true });

    await page.getByRole('link', { name: 'Template review', exact: true }).click();
    await page.getByLabel('Utility draft event').selectOption('invoice');
    const draft = await page.getByLabel('Message body', { exact: true }).inputValue();
    await page.getByLabel('Message body', { exact: true }).fill(draft + ' Upgrade today.');
    await page.locator('#relationship').check();
    await page.getByRole('button', { name: 'Assess template', exact: true }).click();
    await page.getByRole('button', { name: 'Use suggested edit', exact: true }).waitFor();
    await page.screenshot({ path: path.join(output, 'review-desktop.png'), fullPage: true });
    await page.getByRole('button', { name: 'Use suggested edit', exact: true }).click();
    assert.equal(await page.getByLabel('Message body', { exact: true }).inputValue(), draft);

    await page.setViewportSize({ width: 390, height: 844 });
    for (const [name, file] of [['Dataset', 'dataset-mobile.png'], ['Template review', 'review-mobile.png'], ['Model lab', 'model-mobile.png'], ['Error review', 'errors-mobile.png'], ['Experiments', 'experiments-mobile.png']]) {
      await page.getByRole('link', { name, exact: true }).click();
      if (name === 'Error review') await page.getByLabel('Review notes', { exact: true }).waitFor();
      if (name === 'Experiments') await page.getByRole('heading', { name: 'Full chronological holdout' }).waitFor();
      await page.screenshot({ path: path.join(output, file), fullPage: true });
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), `Horizontal page overflow: ${name}`);
    }
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ status: 'passed', imported_records: summary.total, browser_errors: errors, report: report.report, screenshots: output }, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
