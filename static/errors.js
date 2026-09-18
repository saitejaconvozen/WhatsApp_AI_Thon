const errorState = { data: null, selected: null, direction: 'false_utility', status: '', query: '', dirty: false, busy: false };
let errorsRequest = 0, experimentsRequest = 0;
const reasonLabels = { unknown: 'Not assessed', promotion: 'Promotional content', missing_context: 'Missing business context', ambiguous_wording: 'Ambiguous wording', label_question: 'Question for platform team', model_error: 'Model error', other: 'Other' };
const reviewStatuses = { pending: 'Pending', context_needed: 'Context needed', reviewed: 'Reviewed' };
const choiceOptions = (options, selected) => Object.entries(options).map(([value, label]) => `<option value="${value}" ${selected === value ? 'selected' : ''}>${esc(label)}</option>`).join('');

window.addEventListener('beforeunload', event => { if (errorState.dirty) { event.preventDefault(); event.returnValue = ''; } });

async function loadErrors() {
  const request = ++errorsRequest;
  if (errorState.dirty && errorState.data) return;
  $('#app').innerHTML = '<div class="loading">Loading held-out errors...</div>';
  try {
    const data = await api('/api/errors');
    if (state.route !== 'errors' || request !== errorsRequest) return;
    errorState.data = data;
    renderErrors();
  } catch (error) {
    if (state.route === 'errors' && request === errorsRequest) {
      $('#app').innerHTML = heading('Error review', 'Held-out predictions only') + `<div class="notice warning">${esc(error.message)}</div><a class="button" href="#model">Open Model lab</a>`;
    }
  }
}

function filteredErrors() {
  return errorState.data.items.filter(row => (!errorState.direction || row.direction === errorState.direction)
    && (!errorState.status || row.annotation.status === errorState.status)
    && `${row.name} ${row.body}`.toLowerCase().includes(errorState.query.toLowerCase()));
}

function renderErrors() {
  const data = errorState.data;
  const rows = filteredErrors();
  if (!rows.some(row => row.id === errorState.selected)) errorState.selected = rows[0]?.id || null;
  $('#app').innerHTML = heading('Error review', `Evaluation snapshot: ${esc(data.trained_at)}`, `<a class="button" href="/api/error-export">${icon('download')}Export reviews</a>`) +
    `<div class="stats">${stat('False utility', num(data.counts.false_utility), 'Meta marketing / predicted utility', 'triangle-alert', 'amber')}${stat('False marketing', num(data.counts.false_marketing), 'Meta utility / predicted marketing', 'arrow-left-right')}${stat('Reviewed', num(data.counts.reviewed), `Of ${num(data.counts.total)} errors`, 'clipboard-check')}${stat('Source labels', 'Unchanged', 'Annotations stored separately', 'lock-keyhole')}</div>
    <div class="toolbar"><label class="search">${icon('search')}<input id="error-search" aria-label="Search errors" placeholder="Search error templates" value="${esc(errorState.query)}"></label><select id="error-direction" aria-label="Error direction">${choiceOptions({ false_utility: 'False utility', false_marketing: 'False marketing', '': 'Both directions' }, errorState.direction)}</select><select id="error-status" aria-label="Review status filter">${choiceOptions({ '': 'All review statuses', ...reviewStatuses }, errorState.status)}</select></div>
    <div class="error-layout"><section aria-label="Error queue"><div id="error-queue"></div></section><section id="error-detail" aria-label="Selected error"></section></div>`;
  $('#error-direction').onchange = event => { if (allowReviewChange()) { errorState.direction = event.target.value; renderErrors(); } else event.target.value = errorState.direction; };
  $('#error-status').onchange = event => { if (allowReviewChange()) { errorState.status = event.target.value; renderErrors(); } else event.target.value = errorState.status; };
  $('#error-search').onchange = event => { if (allowReviewChange()) { errorState.query = event.target.value; renderErrors(); } else event.target.value = errorState.query; };
  $('#error-queue').innerHTML = rows.length ? rows.map(row => `<button class="error-item ${row.id === errorState.selected ? 'selected' : ''}" data-error-id="${row.id}" aria-pressed="${row.id === errorState.selected}"><strong>${esc(row.name || 'Untitled template')}</strong><span>${esc(row.body.slice(0, 140))}</span><small>${reviewStatuses[row.annotation.status]}</small></button>`).join('') : '<div class="empty"><h2>No matching errors</h2></div>';
  document.querySelectorAll('[data-error-id]').forEach(button => { button.onclick = () => { if (allowReviewChange()) { errorState.selected = button.dataset.errorId; renderErrors(); } }; });
  renderErrorDetail(); icons();
}

function allowReviewChange() {
  if (errorState.busy) { toast('Wait for the current request to finish.'); return false; }
  if (errorState.dirty) { toast('Save or discard this review before changing records.', true); return false; }
  return true;
}

function selectedError() { return errorState.data.items.find(row => row.id === errorState.selected); }

function renderErrorDetail() {
  const row = selectedError();
  if (!row) { $('#error-detail').innerHTML = ''; return; }
  const a = row.annotation;
  const draft = a.draft || row;
  $('#error-detail').innerHTML = `<h2>${esc(row.name || 'Untitled template')}</h2><div class="record-meta"><span>Meta ${badge(row.recorded_category)}</span><span>Predicted ${badge(row.prediction)}</span></div>
    <details open><summary>Original template</summary>${['header', 'body', 'footer', 'buttons'].map(key => `<div class="error-component"><h3>${key[0].toUpperCase() + key.slice(1)}</h3><p>${esc(row[key] || '(empty)')}</p></div>`).join('')}</details>
    <form id="error-form"><div class="field-row"><div class="field"><label for="annotation-status">Review status</label><select id="annotation-status">${choiceOptions(reviewStatuses, a.status)}</select></div><div class="field"><label for="annotation-reason">Review reason</label><select id="annotation-reason">${choiceOptions(reasonLabels, a.reason)}</select></div></div>
    <div class="field"><label for="annotation-notes">Review notes</label><textarea id="annotation-notes" maxlength="10000">${esc(a.notes)}</textarea></div>
    <div class="field"><label for="annotation-context">Business event and recipient context</label><textarea id="annotation-context" maxlength="10000">${esc(a.context)}</textarea></div>
    <div class="field"><label for="annotation-purpose">Service event</label><select id="annotation-purpose">${choiceOptions({ unknown: 'Not established', billing: 'Billing / payment', order: 'Order / delivery', appointment: 'Appointment', support: 'Support case', account: 'Account update', critical: 'Critical alert' }, a.purpose)}</select></div>
    <label class="check"><input id="annotation-relationship" type="checkbox" ${a.relationship_confirmed ? 'checked' : ''}>Confirmed actual transaction, requested service or critical recipient need</label>
    <details class="candidate-editor"><summary>Candidate draft and observed outcome</summary>${['header', 'body', 'footer', 'buttons'].map(key => `<div class="field"><label for="candidate-${key}">Candidate ${key}</label><textarea id="candidate-${key}" maxlength="${key === 'body' ? 20000 : 4000}">${esc(draft[key] || '')}</textarea></div>`).join('')}
    <div class="actions"><button class="button" id="check-candidate" type="button">${icon('scan-text')}Assess candidate</button></div><div id="candidate-result"></div>
    <div class="field"><label for="meta-outcome">Observed Meta outcome</label><select id="meta-outcome">${choiceOptions({ not_submitted: 'Not submitted / not recorded', MARKETING: 'Marketing', UTILITY: 'Utility', AUTHENTICATION: 'Authentication', REJECTED: 'Rejected' }, a.meta_outcome)}</select></div>
    <div class="field"><label for="outcome-reference">Meta decision reference or timestamp</label><input id="outcome-reference" maxlength="1000" value="${esc(a.outcome_reference)}"></div><p class="muted">Manually reported outcome for this exact draft. No submission is sent by this app.</p></details>
    <div class="actions"><button class="button primary" id="save-error" type="submit">${icon('save')}Save review</button><button class="button" id="discard-error" type="button">${icon('undo-2')}Discard edits</button><span id="review-save-state" role="status" class="muted">Saved version ${a.version}</span></div></form>`;
  $('#error-form').oninput = event => {
    errorState.dirty = true; $('#review-save-state').textContent = 'Unsaved changes';
    if (event.target.id.startsWith('candidate-') || ['annotation-purpose', 'annotation-relationship'].includes(event.target.id)) $('#candidate-result').innerHTML = '';
    if (event.target.id.startsWith('candidate-')) {
      $('#meta-outcome').value = 'not_submitted'; $('#outcome-reference').value = '';
    }
  };
  $('#discard-error').onclick = () => { if (!errorState.busy) { errorState.dirty = false; renderErrors(); } };
  $('#error-form').onsubmit = saveError;
  $('#check-candidate').onclick = assessCandidate;
}

function errorPayload() {
  const row = selectedError();
  return { snapshot: errorState.data.snapshot, version: row.annotation.version,
    status: $('#annotation-status').value, reason: $('#annotation-reason').value,
    notes: $('#annotation-notes').value, context: $('#annotation-context').value,
    purpose: $('#annotation-purpose').value, relationship_confirmed: $('#annotation-relationship').checked,
    draft: Object.fromEntries(['header', 'body', 'footer', 'buttons'].map(key => [key, $(`#candidate-${key}`).value])),
    meta_outcome: $('#meta-outcome').value, outcome_reference: $('#outcome-reference').value };
}

async function saveError(event) {
  event.preventDefault();
  if (errorState.busy) return;
  const row = selectedError(), payload = errorPayload();
  errorState.busy = true; $('#save-error').disabled = true; $('#error-form').inert = true;
  try {
    row.annotation = await post(`/api/errors/${row.id}`, payload);
    errorState.dirty = false;
    errorState.data.counts.reviewed = errorState.data.items.filter(r => r.annotation.status === 'reviewed').length;
    if (state.route === 'errors') renderErrors();
    toast('Review saved. Source labels unchanged.');
  } catch (error) { toast(error.message, true); }
  finally { errorState.busy = false; if ($('#save-error')) $('#save-error').disabled = false; if ($('#error-form')) $('#error-form').inert = false; }
}

async function assessCandidate() {
  if (errorState.busy) return;
  const payload = errorPayload(), id = errorState.selected;
  errorState.busy = true; $('#check-candidate').disabled = true; $('#error-form').inert = true;
  try {
    const result = await post('/api/review', { ...payload.draft, purpose: payload.purpose, relationship_confirmed: payload.relationship_confirmed, format: 'TEXT' });
    if (state.route !== 'errors' || id !== errorState.selected) return;
    const policy = result.policy;
    $('#candidate-result').innerHTML = `<div class="notice">${badge(policy.category)}<p>${esc(policy.summary)}</p><p>${esc(policy.rewrite.reason)}</p></div>${policy.rewrite.available ? '<button type="button" class="button" id="apply-candidate">Use limited edit</button>' : ''}`;
    if ($('#apply-candidate')) $('#apply-candidate').onclick = () => {
      Object.entries(policy.rewrite.components).forEach(([key, value]) => { $(`#candidate-${key}`).value = value; });
      $('#meta-outcome').value = 'not_submitted'; $('#outcome-reference').value = '';
      errorState.dirty = true; $('#review-save-state').textContent = 'Unsaved changes';
      $('#candidate-result').innerHTML = '<div class="notice">Limited edit applied. Facts and placeholders still require human review.</div>';
    };
  } catch (error) { toast(error.message, true); }
  finally { errorState.busy = false; if ($('#check-candidate')) $('#check-candidate').disabled = false; if ($('#error-form')) $('#error-form').inert = false; }
}

async function loadExperiments() {
  const request = ++experimentsRequest;
  $('#app').innerHTML = '<div class="loading">Loading experiments...</div>';
  try {
    const data = await api('/api/experiments');
    if (state.route === 'experiments' && request === experimentsRequest) renderExperiments(data);
  } catch (error) { if (state.route === 'experiments') $('#app').innerHTML = `<div class="notice warning">${esc(error.message)}</div>`; }
}

function renderExperiments(data) {
  const r = data.report;
  $('#app').innerHTML = heading('Model experiments', 'Chronological snapshot evaluation / serving model unchanged', `<button class="button primary" id="run-experiments" ${data.running ? 'disabled' : ''}>${icon('flask-conical')}${data.running ? 'Running...' : 'Run comparison'}</button>`) +
    (r ? `${data.stale ? '<div class="notice warning">Dataset changed. Rerun this comparison.</div>' : ''}<div class="stats">${stat('Training families', num(r.split.train_families), 'Before cutoff, including update dates', 'layers')}${stat('Test families', num(r.split.test_families), 'First observed on or after cutoff', 'calendar')}${stat('Boundary exclusions', num(r.split.purged_boundary_families), 'Families spanning the cutoff', 'filter')}${stat('High-overlap test families', num(r.near_duplicates.flagged_test_families), 'Character cosine at least 0.90', 'copy')}</div>
    <p class="source-note">Cutoff: ${esc(r.split.cutoff)}. Invalid-date families excluded: ${num(r.split.excluded_invalid_date_families)}. Family overlap: ${num(r.split.group_overlap)}. Majority accuracy: ${percent(r.majority_accuracy)}.</p>
    ${['all', 'lower_overlap'].map((subset, index) => `<section class="development-section"><h2>${index ? 'Lower lexical overlap subset' : 'Full chronological holdout'}</h2><div class="table-wrap"><table><thead><tr><th>Method</th><th>Examples</th><th>Utility precision</th><th>Utility recall</th><th>False utility</th><th>Accuracy</th></tr></thead><tbody>${r.results.map(result => { const m = result[subset]; return `<tr><th>${esc(result.method)}</th>${m ? `<td>${num(m.samples)}</td><td>${percent(m.utility_precision)}</td><td>${percent(m.utility_recall)}</td><td>${num(m.confusion_matrix[0][1])}</td><td>${percent(m.accuracy)}</td>` : '<td colspan="5">No examples in subset</td>'}</tr>`; }).join('')}</tbody></table></div></section>`).join('')}
    <section class="development-section"><h2>Interpretation limits</h2><ul>${r.limitations.map(text => `<li>${esc(text)}</li>`).join('')}</ul></section>` : '<div class="empty"><h2>No chronological comparison yet</h2><p>Word features, character features and local latent-semantic features; no external API.</p></div>');
  $('#run-experiments').onclick = async event => {
    event.currentTarget.disabled = true; event.currentTarget.textContent = 'Running...';
    try { const result = await post('/api/experiments/run', {}); if (state.route === 'experiments') renderExperiments({ ...result, running: false }); toast('Comparison complete. Serving model unchanged.'); }
    catch (error) { toast(error.message, true); if (state.route === 'experiments') loadExperiments(); }
  };
  icons();
}
