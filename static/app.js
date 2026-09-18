const $ = (selector, root = document) => root.querySelector(selector);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const num = value => Number(value || 0).toLocaleString();
const percent = value => `${(100 * Number(value || 0)).toFixed(1)}%`;
const icon = name => `<i data-lucide="${name}"></i>`;
const labels = { UNKNOWN: 'Unrecorded', MARKETING: 'Marketing', UTILITY: 'Utility', AUTHENTICATION: 'Authentication', VERIFIED: 'Verified', APPROVED: 'Approved', FAILED: 'Failed', REJECTED: 'Rejected', PENDING: 'Pending', IN_PROGRESS: 'In progress', LIKELY_MARKETING: 'Likely marketing', UTILITY_CANDIDATE: 'Utility candidate', NEEDS_REVIEW: 'Needs review' };
const badge = value => `<span class="badge ${esc(String(value).toLowerCase())}">${esc(labels[value] || value)}</span>`;
const icons = () => window.lucide?.createIcons();
const state = { summary: null, model: null, presets: {}, page: 1, query: '', category: '', status: '', mismatch: false, route: '', review: { header: '', body: '', footer: '', buttons: '', purpose: 'unknown', relationship_confirmed: false, format: 'TEXT' }, result: null, source: null, training: false };
let toastTimer, queryTimer, listVersion = 0;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    const detail = typeof error.detail === 'string' ? error.detail : 'The request could not be completed.';
    throw new Error(detail);
  }
  return response.json();
}
const post = (path, payload) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });

function toast(message, error = false) {
  clearTimeout(toastTimer);
  const element = $('#toast');
  element.textContent = message;
  element.classList.toggle('error', error);
  element.hidden = false;
  toastTimer = setTimeout(() => { element.hidden = true; }, error ? 9000 : 4500);
}

async function refresh() {
  const [summary, model] = await Promise.all([api('/api/summary'), api('/api/model')]);
  state.summary = summary;
  state.model = model;
  $('#nav-count').textContent = num(summary.total);
}

function heading(title, subtitle, actions = '') {
  return `<div class="page-heading"><div><h1>${title}</h1><p class="subtitle">${subtitle}</p></div><div class="actions">${actions}</div></div>`;
}

function stat(label, value, note, symbol, tone = '') {
  return `<div class="stat"><div class="label">${icon(symbol)}${label}</div><div class="value ${tone}">${value}</div><div class="note">${note}</div></div>`;
}

function navigate() {
  if (state.route === 'errors' && errorState.dirty && location.hash !== '#errors') {
    location.hash = '#errors';
    toast('Save or discard your error review before leaving.', true);
    return;
  }
  state.route = ['development', 'dataset', 'compose', 'review', 'model', 'errors', 'experiments', 'benchmark'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'development';
  document.querySelectorAll('[data-nav]').forEach(a => a.classList.toggle('active', a.dataset.nav === state.route));
  $('#breadcrumb-page').textContent = { development: 'Development', dataset: 'Dataset', compose: 'Compose', review: 'Template review', model: 'Model lab', errors: 'Error review', experiments: 'Experiments', benchmark: 'DeepSeek evaluation' }[state.route];
  if (state.route === 'development') renderDevelopment();
  if (state.route === 'dataset') renderDataset();
  if (state.route === 'compose') renderCompose();
  if (state.route === 'review') renderReview();
  if (state.route === 'model') renderModel();
  if (state.route === 'errors') loadErrors();
  if (state.route === 'experiments') loadExperiments();
  if (state.route === 'benchmark') renderBenchmark();
  icons();
}

function renderDataset() {
  const s = state.summary;
  const distribution = [['MARKETING', '#d6a355'], ['UTILITY', '#55a18a'], ['AUTHENTICATION', '#7995c5'], ['UNKNOWN', '#cdd5cf']];
  $('#app').innerHTML = heading('Template dataset', `${num(s.total)} records across ${num(s.families)} normalized body families`, `<a class="button" href="/api/export" title="Export dataset">${icon('download')}Export</a><button class="button primary" id="open-import">${icon('plus')}Import templates</button>`) +
    `<div class="stats">${stat('Total templates', num(s.total), `${num(Object.keys(s.sources).length)} imported file(s)`, 'files')}${stat('Recorded utility', num(s.categories.UTILITY), 'Across all workflow statuses', 'check-check', 'teal')}${stat('Utility to marketing', num(s.category_mismatches), 'Requested / recorded disagreement', 'arrow-left-right', 'amber')}${stat('Training families', num(s.training_families), 'Eligible, distinct text examples', 'layers')}</div>
    <div class="overview"><section><div class="section-heading"><h2>Recorded categories</h2><span class="muted">All records</span></div><div class="distribution-bar" role="img" aria-label="${esc(distribution.map(([key]) => `${labels[key]}: ${num(s.categories[key])}`).join(', '))}">${distribution.map(([key, color]) => `<span style="width:${s.total ? (s.categories[key] || 0) / s.total * 100 : 0}%;background:${color}"></span>`).join('')}</div><div class="legend">${distribution.map(([key, color]) => `<span><i class="swatch" style="background:${color}"></i>${labels[key]} <strong>${num(s.categories[key])}</strong></span>`).join('')}</div></section>
    <section><div class="section-heading"><h2>Data quality</h2><span class="muted">Before training</span></div><div class="quality-list"><div class="quality-item"><span>Unrecorded categories</span><strong>${num(s.categories.UNKNOWN)}</strong></div><div class="quality-item"><span>Repeated body families</span><strong>${num(s.duplicate_records)}</strong></div><div class="quality-item"><span>Conflicting labeled families</span><strong>${num(s.conflicting_families)}</strong></div><div class="quality-item"><span>Missing message bodies</span><strong>${num(s.missing_bodies)}</strong></div></div></section></div>
    <div class="section-heading"><h2>All templates</h2><span id="result-count" class="muted"></span></div>
    <div class="toolbar"><label class="search">${icon('search')}<input id="search" aria-label="Search templates" placeholder="Search names or message text" value="${esc(state.query)}"></label><select id="category-filter" aria-label="Filter by recorded category"><option value="">All categories</option>${['MARKETING', 'UTILITY', 'AUTHENTICATION', 'UNKNOWN'].map(v => `<option value="${v}" ${state.category === v ? 'selected' : ''}>${labels[v]}</option>`).join('')}</select><select id="status-filter" aria-label="Filter by status"><option value="">All statuses</option>${Object.keys(s.statuses).map(v => `<option value="${esc(v)}" ${state.status === v ? 'selected' : ''}>${esc(labels[v] || v)}</option>`).join('')}</select><label class="check"><input type="checkbox" id="mismatch-filter" ${state.mismatch ? 'checked' : ''}>Category disagreements</label><button class="icon-button" id="reset-filters" title="Clear filters" aria-label="Clear filters">${icon('filter-x')}</button></div><div id="record-table"></div>`;
  $('#open-import').onclick = openImport;
  $('#search').oninput = event => {
    state.query = event.target.value;
    state.page = 1;
    clearTimeout(queryTimer);
    queryTimer = setTimeout(loadRecords, 250);
  };
  $('#category-filter').onchange = event => { state.category = event.target.value; state.page = 1; loadRecords(); };
  $('#status-filter').onchange = event => { state.status = event.target.value; state.page = 1; loadRecords(); };
  $('#mismatch-filter').onchange = event => { state.mismatch = event.target.checked; state.page = 1; loadRecords(); };
  $('#reset-filters').onclick = () => { Object.assign(state, { query: '', category: '', status: '', mismatch: false, page: 1 }); renderDataset(); icons(); };
  loadRecords();
}

async function loadRecords() {
  const version = ++listVersion;
  if (state.route !== 'dataset') return;
  const query = new URLSearchParams({ q: state.query, category: state.category, status: state.status, mismatch: String(state.mismatch), page: state.page });
  try {
    const data = await api(`/api/records?${query}`);
    if (version !== listVersion || state.route !== 'dataset') return;
    $('#result-count').textContent = `${num(data.total)} results`;
    if (!data.items.length) {
      $('#record-table').innerHTML = `<div class="table-wrap empty">${icon(state.summary.total ? 'search-x' : 'files')}<h2>${state.summary.total ? 'No matching templates' : 'No templates imported'}</h2><p>${state.summary.total ? 'No records match the current filters.' : 'Your dataset will appear here.'}</p>${state.summary.total ? '' : '<button class="button primary" id="empty-import">Import templates</button>'}</div>`;
      if ($('#empty-import')) $('#empty-import').onclick = openImport;
      icons();
      return;
    }
    const totalPages = Math.ceil(data.total / data.page_size);
    $('#record-table').innerHTML = `<div class="table-wrap"><table class="data-table"><thead><tr><th>Template</th><th>Requested</th><th>Recorded</th><th>Status</th><th>Format</th><th></th></tr></thead><tbody>${data.items.map(r => `<tr><td><button class="template-name" data-record="${r.id}" title="${esc(r.name)}">${esc(r.name || 'Untitled template')}</button><span class="excerpt">${esc(r.body || 'No message body')}</span></td><td>${badge(r.requested_category)}</td><td>${badge(r.meta_category)}</td><td><span class="status-text">${esc(labels[r.status] || r.status)}</span></td><td><span class="status-text">${esc(r.format)}</span></td><td><button class="icon-button" data-record="${r.id}" aria-label="Open ${esc(r.name || 'template')}" title="Open template">${icon('arrow-up-right')}</button></td></tr>`).join('')}</tbody></table></div><div class="pagination"><span>Showing ${num((data.page - 1) * data.page_size + 1)}–${num(Math.min(data.page * data.page_size, data.total))} of ${num(data.total)}</span><div class="actions"><span>Page ${num(data.page)} of ${num(totalPages)}</span><button class="icon-button" id="previous-page" title="Previous page" aria-label="Previous page" ${state.page <= 1 ? 'disabled' : ''}>${icon('chevron-left')}</button><button class="icon-button" id="next-page" title="Next page" aria-label="Next page" ${state.page >= totalPages ? 'disabled' : ''}>${icon('chevron-right')}</button></div></div>`;
    $('#previous-page').onclick = () => { state.page--; loadRecords(); };
    $('#next-page').onclick = () => { state.page++; loadRecords(); };
    document.querySelectorAll('[data-record]').forEach(button => { button.onclick = () => openRecord(button.dataset.record); });
    icons();
  } catch (error) {
    if (state.route === 'dataset') $('#record-table').innerHTML = `<div class="notice warning">${esc(error.message)}</div>`;
    toast(error.message, true);
  }
}

async function openRecord(id) {
  try {
    const record = await api(`/api/records/${encodeURIComponent(id)}`);
    $('#record-title').textContent = record.name || 'Template details';
    $('#record-content').innerHTML = `<div class="record-meta"><span>Requested ${badge(record.requested_category)}</span><span>Recorded ${badge(record.meta_category)}</span><span>${esc(record.format)} · ${esc(labels[record.status] || record.status)}</span></div>${record.header ? `<h3 style="margin-top:20px">${esc(record.header)}</h3>` : ''}<div class="record-body">${esc(record.body || 'No message body')}</div>${record.footer ? `<p class="muted">${esc(record.footer)}</p>` : ''}${record.buttons ? `<div class="draft-block">${esc(record.buttons)}</div>` : ''}<div class="detail-list"><div><span>Language</span><span>${esc(record.language || 'Unspecified')}</span></div><div><span>Created</span><span>${esc(record.created_at.slice(0, 10) || 'Unrecorded')}</span></div><div><span>Recorded label field</span><span>${esc(record.label_source)}</span></div></div><div class="dialog-actions"><button class="button primary" id="review-record">${icon('scan-text')}Review template</button></div>`;
    $('#review-record').onclick = () => {
      state.review = { header: record.header, body: record.body, footer: record.footer, buttons: record.buttons, purpose: 'unknown', relationship_confirmed: false, format: record.format };
      state.source = record;
      state.result = null;
      $('#record-dialog').close();
      if (location.hash === '#review') { renderReview(); icons(); } else location.hash = '#review';
    };
    $('#record-dialog').showModal();
    icons();
  } catch (error) { toast(error.message, true); }
}

function openImport() {
  $('#import-content').innerHTML = `<div class="dropzone" id="dropzone">${icon('upload')}<strong>Select a template export</strong><p class="muted">JSON, JSONL, CSV, TSV or XLSX · Maximum 25 MB</p><label class="button primary" for="upload-file">${icon('folder-open')}Choose file</label><input type="file" id="upload-file" accept=".json,.jsonl,.csv,.tsv,.xlsx,.txt" hidden></div><div class="dialog-actions"><a class="button" href="/api/schema">${icon('download')}CSV columns</a></div>`;
  $('#upload-file').onchange = event => previewUpload(event.target.files[0]);
  const zone = $('#dropzone');
  zone.ondragover = event => { event.preventDefault(); zone.classList.add('dragover'); };
  zone.ondragleave = () => zone.classList.remove('dragover');
  zone.ondrop = event => { event.preventDefault(); previewUpload(event.dataTransfer.files[0]); };
  if (!$('#import-dialog').open) $('#import-dialog').showModal();
  icons();
}

async function previewUpload(file) {
  if (!file) return;
  if (file.size > 25 * 1024 * 1024) return toast('The file exceeds the 25 MB limit.', true);
  $('#import-content').innerHTML = '<div class="loading">Reading template export…</div>';
  const form = new FormData();
  form.append('file', file);
  try {
    const result = await api('/api/import/preview', { method: 'POST', body: form });
    const fieldNames = { name: 'Template name', body: 'Message body', header: 'Header', footer: 'Footer', buttons: 'Button text', meta_category: 'Recorded Meta category', requested_category: 'Requested category', status: 'Verification status', language: 'Language', created_at: 'Created date', family_id: 'Template family ID' };
    $('#import-content').innerHTML = `<div class="section-heading"><h3>${esc(file.name)}</h3><span class="badge utility">${num(result.count)} records</span></div>${result.nested ? '<div class="notice">Nested draft export detected. Recorded category: <code>metaTemplateCategory</code>. Requested category: <code>messageBody.templateCategory</code>.</div>' : `<div class="mapping-grid">${result.fields.map(field => `<div class="field"><label for="map-${field}">${fieldNames[field]}</label><select id="map-${field}" data-map="${field}"><option value="">Not supplied</option>${result.columns.map(column => `<option value="${esc(column)}" ${result.mapping[field] === column ? 'selected' : ''}>${esc(column)}</option>`).join('')}</select></div>`).join('')}</div><div class="notice">Map the recorded category only to actual Meta decisions. Missing labels and unfinished statuses are excluded from training.</div>`}<div class="import-preview"><h3>File preview</h3>${result.preview.slice(0, 3).map(r => `<p><strong>${esc(r.name || 'Untitled')}</strong><br>${esc(r.body.slice(0, 160) || 'Message body column not yet mapped')}</p>`).join('')}</div><div class="dialog-actions"><button class="button" id="choose-another">Choose another file</button><button class="button primary" id="commit-import">${icon('database')}Import ${num(result.count)} records</button></div>`;
    $('#choose-another').onclick = openImport;
    $('#commit-import').onclick = async () => {
      const mapping = { ...result.mapping };
      document.querySelectorAll('[data-map]').forEach(select => { mapping[select.dataset.map] = select.value; });
      const button = $('#commit-import');
      button.disabled = true;
      button.textContent = 'Importing…';
      try {
        const imported = await post('/api/import/commit', { token: result.token, mapping });
        $('#import-dialog').close();
        await refresh();
        state.page = 1;
        navigate();
        toast(`${num(imported.added)} records imported. ${num(imported.already_present)} already present.`);
      } catch (error) { toast(error.message, true); button.disabled = false; button.textContent = 'Retry import'; }
    };
    icons();
  } catch (error) { openImport(); toast(error.message, true); }
}

function renderReview() {
  const r = state.review;
  $('#app').innerHTML = heading('Template review', state.source ? `Source snapshot: ${esc(state.source.name)}` : 'Working draft', `<select id="draft-preset" aria-label="Utility draft event"><option value="">Utility draft event</option>${Object.entries(state.presets).map(([key, value]) => `<option value="${key}">${esc(value.name)}</option>`).join('')}</select><button class="icon-button" id="clear-draft" title="Clear draft" aria-label="Clear draft">${icon('rotate-ccw')}</button>`) +
    `${state.source ? `<div class="source-note">Original record: requested ${badge(state.source.requested_category)} · recorded ${badge(state.source.meta_category)} · ${esc(state.source.format)}. The source snapshot is unchanged by draft edits.</div>` : ''}
    <div class="review-layout"><form id="review-form" class="review-editor"><div class="field"><label for="message-body">Message body</label><textarea id="message-body" class="editor-body" maxlength="20000" required placeholder="Enter your template message…">${esc(r.body)}</textarea><div class="field-meta"><span id="character-count">${num(r.body.length)} characters</span><span>${esc(r.format)} template</span></div></div>
    <details ${r.header || r.footer || r.buttons ? 'open' : ''}><summary>Header, footer &amp; buttons</summary><div class="field"><label for="message-header">Header</label><input id="message-header" value="${esc(r.header)}" maxlength="4000"></div><div class="field"><label for="message-footer">Footer</label><input id="message-footer" value="${esc(r.footer)}" maxlength="4000"></div><div class="field"><label for="message-buttons">Button labels</label><textarea id="message-buttons" rows="2" maxlength="4000" placeholder="View invoice">${esc(r.buttons)}</textarea></div></details>
    <div class="field"><label for="message-purpose">Triggering event</label><select id="message-purpose">${[['unknown', 'Not specified'], ['billing', 'Billing or payment event'], ['order', 'Order or delivery update'], ['appointment', 'Appointment or reservation'], ['support', 'Requested support update'], ['account', 'Existing account or service update'], ['critical', 'Critical safety information'], ['promotion', 'Promotion or purchase campaign']].map(([key, label]) => `<option value="${key}" ${r.purpose === key ? 'selected' : ''}>${label}</option>`).join('')}</select></div><label class="check"><input id="relationship" type="checkbox" ${r.relationship_confirmed ? 'checked' : ''}><span>This relates to a real transaction, requested service, or critical need of the recipient.</span></label><div class="actions" style="margin-top:22px"><button class="button primary" type="submit" id="assess-button">${icon('scan-text')}Assess template</button><button class="button" type="button" id="copy-draft">${icon('copy')}Copy draft</button></div></form>
    <section class="preview" aria-label="Message preview"><div class="preview-header"><span class="preview-avatar">${icon('building-2')}</span><div><strong>Business message</strong><small>Template preview</small></div></div><div class="preview-content"><div class="message"><strong id="preview-header"></strong><div id="preview-body"></div><div class="message-footer" id="preview-footer"></div><div id="preview-buttons"></div><div class="message-time">12:00</div></div></div><div class="preview-label">Placeholder values are not filled in</div></section></div><section class="results" id="review-results"></section>`;
  $('#draft-preset').onchange = event => {
    const preset = state.presets[event.target.value];
    if (!preset) return;
    state.review = { header: '', footer: '', body: preset.body, buttons: preset.buttons, purpose: preset.purpose, relationship_confirmed: false, format: 'TEXT' };
    state.source = null;
    state.result = null;
    renderReview(); icons();
  };
  $('#clear-draft').onclick = () => {
    state.review = { header: '', body: '', footer: '', buttons: '', purpose: 'unknown', relationship_confirmed: false, format: 'TEXT' };
    state.source = null; state.result = null; renderReview(); icons();
  };
  for (const [key, selector] of Object.entries({ body: '#message-body', header: '#message-header', footer: '#message-footer', buttons: '#message-buttons', purpose: '#message-purpose' })) {
    $(selector).addEventListener('input', event => { state.review[key] = event.target.value; state.result = null; updatePreview(); renderResults(); });
  }
  $('#relationship').onchange = event => { state.review.relationship_confirmed = event.target.checked; state.result = null; renderResults(); };
  $('#copy-draft').onclick = () => copyText([state.review.header, state.review.body, state.review.footer, state.review.buttons].filter(Boolean).join('\n\n'));
  $('#review-form').onsubmit = async event => {
    event.preventDefault();
    const button = $('#assess-button');
    button.disabled = true;
    button.textContent = 'Assessing…';
    const snapshot = JSON.stringify(state.review);
    try {
      const result = await post('/api/review', state.review);
      if (snapshot === JSON.stringify(state.review)) state.result = result;
      if (state.route === 'review') renderResults();
    } catch (error) { toast(error.message, true); }
    finally { button.disabled = false; button.innerHTML = `${icon('scan-text')}Assess template`; icons(); }
  };
  updatePreview();
  renderResults();
}

function updatePreview() {
  $('#preview-header').textContent = state.review.header;
  $('#preview-header').hidden = !state.review.header;
  $('#preview-body').textContent = state.review.body || 'Your message will appear here.';
  $('#preview-footer').textContent = state.review.footer;
  $('#preview-buttons').innerHTML = state.review.buttons.split('\n').filter(Boolean).map(text => `<div class="message-button">${esc(text)}</div>`).join('');
  $('#character-count').textContent = `${num(state.review.body.length)} characters`;
}

function renderResults() {
  if (!state.result) {
    $('#review-results').innerHTML = '<div class="section-heading"><h2>Assessment</h2><span class="muted">No current assessment</span></div>';
    return;
  }
  const { policy: p, prediction: m } = state.result;
  $('#review-results').innerHTML = `<div class="result-grid"><section><div class="result-title"><h2>Policy checklist</h2>${badge(p.category)}</div><p class="result-summary">${esc(p.summary)}</p><div class="findings">${p.findings.map(f => `<div class="finding">${icon('flag')}<div><code>${esc(f.evidence)}</code> <span class="muted">${esc(f.component)}</span><small>${esc(f.message)}</small></div></div>`).join('')}</div>${p.missing_context.length ? `<h3>Context to verify</h3><ul class="note-list">${p.missing_context.map(text => `<li>${esc(text)}</li>`).join('')}</ul>` : ''}<p class="muted">${esc(p.limitation)}</p><p style="margin-top:10px"><a class="muted" href="${esc(p.policy_url)}" target="_blank" rel="noopener noreferrer">Meta category definitions ${icon('external-link')}</a></p></section>
    <section><div class="result-title"><h2>Model prediction</h2>${m.available ? badge(m.category) : '<span class="badge">Unavailable</span>'}</div>${m.available ? `<p class="result-summary">${esc(m.method)} using historical recorded categories.</p>${m.seen_family ? '<div class="notice warning">This body family is in the training data. This prediction is not an independent evaluation.</div>' : ''}<p class="muted">${esc(m.limitation)}</p>` : `<p class="result-summary muted">${esc(m.reason)}</p><a class="button" href="#model">${icon('chart-no-axes-combined')}Model lab</a>`}</section></div>
    <div class="limitations"><div class="section-heading"><h2>Suggested edit</h2><span class="muted">Limited sentence removal</span></div><p class="muted">${esc(p.rewrite.reason)}</p>${p.rewrite.available ? `<div class="draft-block">${esc([p.rewrite.components.header, p.rewrite.components.body, p.rewrite.components.footer, p.rewrite.components.buttons].filter(Boolean).join('\n\n'))}</div><h3>Removed content</h3><ul class="note-list">${p.rewrite.removed.map(item => `<li>${esc(item.component)}: ${esc(item.text)}</li>`).join('')}</ul><button class="button" id="apply-rewrite">${icon('check')}Use suggested edit</button>` : ''}</div>
    ${m.neighbors?.length ? `<div class="neighbors"><h2>Similar labeled templates</h2>${m.neighbors.map(n => `<div class="neighbor"><div class="neighbor-head"><button class="template-name" data-neighbor="${n.id}">${esc(n.name || 'Untitled template')}</button>${badge(n.category)}</div><p>${esc(n.body.slice(0, 300))}${n.body.length > 300 ? '…' : ''}</p></div>`).join('')}</div>` : ''}`;
  if ($('#apply-rewrite')) $('#apply-rewrite').onclick = () => {
    Object.assign(state.review, p.rewrite.components);
    state.result = null;
    renderReview(); icons(); toast('Suggested edit applied. Assess the revised draft before use.');
  };
  document.querySelectorAll('[data-neighbor]').forEach(button => { button.onclick = () => openRecord(button.dataset.neighbor); });
  icons();
}

async function copyText(text) {
  try { await navigator.clipboard.writeText(text); toast('Draft copied.'); }
  catch { toast('Clipboard access is unavailable in this browser.', true); }
}

function renderModel() {
  const m = state.model;
  const r = m.report;
  const busy = state.training;
  $('#app').innerHTML = heading('Model lab', 'Text category baseline', `${m.trained ? `<a class="button" href="/api/model/report">${icon('download')}Export report</a>` : ''}<button id="train-model" class="button primary ${busy ? 'spin' : ''}" ${busy ? 'disabled' : ''}>${icon(busy ? 'loader-circle' : 'play')}${busy ? 'Training baseline…' : m.trained ? 'Retrain baseline' : 'Train baseline'}</button>`) +
    `${m.stale ? '<div class="notice warning">The dataset changed after training. Retrain to refresh predictions and evaluation.</div>' : ''}
    ${m.trained ? `<div class="stats">${stat('Utility precision', percent(r.per_class.UTILITY.precision), `${num(r.predicted_utility_count)} utility predictions in holdout`, 'target', 'teal')}${stat('Utility recall', percent(r.per_class.UTILITY.recall), `${num(r.per_class.UTILITY.support)} actual utility examples`, 'scan-line')}${stat('Macro F1', percent(r.macro_f1), 'Equal weight for both categories', 'chart-no-axes-combined')}${stat('Test families', num(r.test_families), `${num(r.train_families)} evaluation training families`, 'layers')}</div>
    <div class="model-sections"><section><div class="section-heading"><h2>Holdout confusion matrix</h2><span class="muted">Columns: predicted</span></div><div class="table-wrap"><table class="matrix"><thead><tr><th>Actual category</th><th>Marketing</th><th>Utility</th></tr></thead><tbody><tr><th>Marketing</th><td class="correct">${num(r.confusion_matrix[0][0])}</td><td class="incorrect">${num(r.confusion_matrix[0][1])}</td></tr><tr><th>Utility</th><td class="incorrect">${num(r.confusion_matrix[1][0])}</td><td class="correct">${num(r.confusion_matrix[1][1])}</td></tr></tbody></table></div></section><section><div class="section-heading"><h2>Evaluation snapshot</h2><span class="badge utility">Trained</span></div><div class="detail-list"><div><span>Method</span><span>${esc(r.method)}</span></div><div><span>Accuracy</span><strong>${percent(r.accuracy)}</strong></div><div><span>Majority-class baseline</span><span>${percent(r.majority_accuracy)}</span></div><div><span>Family overlap across splits</span><span>${num(r.group_overlap)}</span></div><div><span>Serving model families</span><span>${num(r.total_families)}</span></div><div><span>Trained</span><span>${esc(new Date(r.trained_at).toLocaleString())}</span></div></div></section></div>
    <div class="limitations"><h2>Evaluation scope</h2><p class="muted" style="margin-top:10px">${esc(r.split)}</p><ul class="note-list">${r.limitations.map(text => `<li>${esc(text)}</li>`).join('')}</ul></div>` : `<div class="stats">${stat('Eligible families', num(state.summary.training_families), 'Completed, labeled text templates', 'layers')}${stat('Marketing families', num(state.summary.training_categories.MARKETING), 'Recorded marketing labels', 'megaphone', 'amber')}${stat('Utility families', num(state.summary.training_categories.UTILITY), 'Recorded utility labels', 'check-check', 'teal')}${stat('Conflicting families', num(state.summary.conflicting_families), 'Excluded from training', 'triangle-alert')}</div><div class="empty">${icon('chart-no-axes-combined')}<h2>No baseline trained yet</h2><p class="muted">Training requires 30 eligible families, with at least 5 in each category.</p></div>`}`;
  $('#train-model').onclick = async () => {
    state.training = true; renderModel(); icons();
    try {
      state.model = await post('/api/model/train', {});
      toast('Baseline trained and evaluated.');
    } catch (error) { toast(error.message, true); }
    finally { state.training = false; if (state.route === 'model') { renderModel(); icons(); } }
  };
}

async function boot() {
  $('#app').innerHTML = '<div class="loading">Opening workspace…</div>';
  $('#close-import').onclick = () => $('#import-dialog').close();
  $('#close-record').onclick = () => $('#record-dialog').close();
  try {
    await refresh();
    state.presets = await api('/api/presets');
    window.addEventListener('hashchange', navigate);
    navigate();
  } catch (error) {
    $('#app').innerHTML = `<div class="empty">${icon('unplug')}<h2>Workspace unavailable</h2><p>${esc(error.message)}</p><button class="button" id="retry-boot">Retry</button></div>`;
    $('#retry-boot').onclick = boot;
    icons();
  }
}

boot();
