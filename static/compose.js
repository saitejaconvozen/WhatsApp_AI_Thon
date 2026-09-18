const composeState = {
  mode: 'inspect',
  draft: { name: '', header: '', body: '', footer: '', buttons: '',
           requested_category: 'UNKNOWN', meta_category: 'UNKNOWN' },
  task: '',
  context: '',
  purpose: '',
  result: null,
  generated: null,
  busy: false,
  source: null,
  relationship: false,
};

const VERDICTS = {
  AUTHENTICATION: ['Authentication template', 'rose', 'key-round'],
  ALREADY_UTILITY: ['Reads as utility already', 'teal', 'circle-check'],
  CONVERTIBLE: ['Can be converted to utility', 'amber', 'wand-sparkles'],
  SPLIT_RECOMMENDED: ['Split into two templates', 'amber', 'split'],
  IRREDUCIBLY_MARKETING: ['Cannot become utility', 'rose', 'circle-slash'],
  NEEDS_CONTEXT: ['Needs more context', 'rose', 'circle-help'],
};

function composeField(key, label, placeholder, rows = 0) {
  const value = esc(composeState.draft[key]);
  return rows
    ? `<label class="field"><span>${label}</span><textarea id="compose-${key}" rows="${rows}" placeholder="${esc(placeholder)}">${value}</textarea></label>`
    : `<label class="field"><span>${label}</span><input id="compose-${key}" value="${value}" placeholder="${esc(placeholder)}"></label>`;
}

function probabilityRow(before, after) {
  if (!before?.available) return `<p class="muted small">${esc(before?.reason || 'The model could not score this template.')}</p>`;
  const cell = (label, value) => `<div><dt>${label}</dt><dd>${value.available ? percent(value.utility_probability) : 'n/a'}${value.available ? ` <span class="muted">${esc(value.band || '')}</span>` : ''}</dd></div>`;
  return `<dl class="facts compact">${cell('Utility probability now', before)}${after ? cell('After the change', after) : ''}</dl>`;
}

function templateCard(title, template, note = '') {
  const line = (label, value) => value ? `<p><strong>${label}:</strong> ${esc(value)}</p>` : '';
  return `<div class="template-card"><h3>${esc(title)}</h3>
    ${line('Header', template.header)}
    <p class="body-text">${esc(template.body)}</p>
    ${line('Footer', template.footer)}
    ${line('Buttons', template.buttons)}
    ${note ? `<p class="muted small">${esc(note)}</p>` : ''}</div>`;
}

function renderCompose() {
  const tab = (key, label) => `<button class="tab ${composeState.mode === key ? 'active' : ''}" data-mode="${key}">${esc(label)}</button>`;
  $('#app').innerHTML = heading(
    'Compose',
    'Check a template and convert it toward utility, or draft a new one from a task',
    `<span class="muted small">Utility rates are far cheaper than marketing</span>`) +
    `<div class="tabs">${tab('inspect', 'Check an existing template')}${tab('generate', 'Generate a new template')}</div>
     <div id="compose-panel"></div>`;
  renderComposePanel();
  icons();
}

function renderComposePanel() {
  if (state.route !== 'compose') return;
  document.querySelectorAll('[data-mode]').forEach(button => {
    button.classList.toggle('active', button.dataset.mode === composeState.mode);
    button.setAttribute('aria-pressed', String(button.dataset.mode === composeState.mode));
  });
  $('#compose-panel').innerHTML = composeState.mode === 'inspect' ? inspectPanel() : generatePanel();
  wireCompose();
  icons();
}

function inspectPanel() {
  const result = composeState.result;
  return `<div class="overview compose-grid">
    <section>
      <div class="section-heading"><h2>Template</h2><span class="muted">Paste it, or upload a file</span></div>
      <div class="upload-row">
        <input type="file" id="compose-file" accept=".json,.jsonl,.csv,.tsv,.xlsx" hidden>
        <button class="button" id="compose-upload">${icon('upload')}Upload a template file</button>
        ${composeState.source ? `<span class="muted small">${esc(composeState.source)}</span>` : ''}
      </div>
      ${composeField('header', 'Header', 'Optional')}
      ${composeField('body', 'Body', 'Your order {{order_id}} has shipped.', 5)}
      ${composeField('footer', 'Footer', 'Optional')}
      ${composeField('buttons', 'Buttons', 'One per line', 2)}
      <label class="field"><span>Category submitted to Meta</span><select id="compose-requested-category">
        ${['UNKNOWN', 'UTILITY', 'MARKETING'].map(c => `<option value="${c}" ${composeState.draft.requested_category === c ? 'selected' : ''}>${c === 'UNKNOWN' ? 'Not specified' : c}</option>`).join('')}
      </select></label>
      <label class="field"><span>Category recorded by Meta</span><select id="compose-meta-category">
        ${['UNKNOWN', 'MARKETING', 'UTILITY', 'AUTHENTICATION'].map(c => `<option value="${c}" ${composeState.draft.meta_category === c ? 'selected' : ''}>${c === 'UNKNOWN' ? 'Not decided / unknown' : c}</option>`).join('')}
      </select></label>
      <label class="check"><input type="checkbox" id="compose-relationship" ${composeState.relationship ? 'checked' : ''}>Confirmed actual transaction or requested service for this recipient</label>
      <button class="button primary" id="compose-check" ${composeState.busy ? 'disabled' : ''}>${icon('scan-text')}${composeState.busy ? 'Checking...' : 'Check this template'}</button>
    </section>
    <section>
      <div class="section-heading"><h2>Verdict</h2>${result ? `<span class="muted">${esc(result.method)}</span>` : ''}</div>
      ${result ? verdictPanel(result) : '<div class="empty">No template checked yet.</div>'}
    </section></div>`;
}

function verdictPanel(result) {
  const [label, tone, mark] = VERDICTS[result.verdict] || ['Unknown', 'gray', 'circle'];
  const offer = result.utility ? `
    <div class="notice">${icon('help-circle')} Candidate service-only edit. Human review and Meta categorization are still required.</div>
    ${templateCard('Utility version', result.utility)}
    ${result.split_off ? templateCard('Promotional part, as a separate marketing template', { body: result.split_off.body }, result.split_off.note) : ''}
    ${result.needs_human ? `<div class="notice warn">${icon('triangle-alert')} Some promotional wording carries business data and was kept. A person has to decide on it:<ul>${result.ambiguous.map(a => `<li>${esc(a.text)}</li>`).join('')}</ul></div>` : ''}
    <div class="actions">
      <button class="button primary" id="compose-accept">${icon('check')}Use the utility version</button>
      <button class="button" id="compose-keep">${icon('x')}Keep it as it is</button>
    </div>` : '';
  return `<h3>Predicted Meta category</h3><p>${result.before?.available ? badge(result.before.category) : esc(result.before?.reason || 'Prediction unavailable')}</p><h3 style="margin-top:20px">Conversion assessment</h3><div class="verdict ${tone}">${icon(mark)}<div><strong>${esc(label)}</strong><p>${esc(result.reason || '')}</p></div></div>
    ${probabilityRow(result.before, result.after)}
    ${result.removed?.length ? `<details><summary>Removed ${result.removed.length} promotional fragment(s)</summary><ul>${result.removed.map(r => `<li><span class="muted">${esc(r.component)}</span> ${esc(r.text)}</li>`).join('')}</ul></details>` : ''}
    ${offer}
    <p class="muted small">${esc(result.limitation || 'Meta decides the final category. This is a local assessment.')}</p>`;
}

function generatePanel() {
  const g = composeState.generated;
  return `<div class="overview compose-grid">
    <section>
      <div class="section-heading"><h2>What do you need to tell the customer?</h2><span class="muted">Utility only</span></div>
      <label class="field"><span>Task</span><textarea id="compose-task" rows="4" placeholder="Tell the customer their invoice is ready and when it is due">${esc(composeState.task)}</textarea></label>
      <label class="field"><span>Business context</span><textarea id="compose-context" rows="3" maxlength="6000" placeholder="Existing transaction, recipient relationship and facts to preserve">${esc(composeState.context)}</textarea></label>
      <label class="field"><span>Service event</span><select id="compose-purpose">
        ${['', 'billing', 'order', 'appointment', 'support', 'account', 'critical'].map(p =>
          `<option value="${p}" ${composeState.purpose === p ? 'selected' : ''}>${p ? p[0].toUpperCase() + p.slice(1) : 'Detect automatically'}</option>`).join('')}
      </select></label>
      <button class="button primary" id="compose-generate" ${composeState.busy ? 'disabled' : ''}>${icon('wand-sparkles')}${composeState.busy ? 'Drafting...' : 'Draft a utility template'}</button>
    </section>
    <section>
      <div class="section-heading"><h2>Draft</h2>${g ? `<span class="muted">${esc(g.method)}</span>` : ''}</div>
      ${g ? generatedPanel(g) : '<div class="empty">No draft yet.</div>'}
    </section></div>`;
}

function generatedPanel(g) {
  if (!g.possible) {
    const [label, tone, mark] = VERDICTS[g.verdict] || ['Not possible', 'rose', 'circle-slash'];
    return `<div class="verdict ${tone}">${icon(mark)}<div><strong>${esc(label)}</strong><p>${esc(g.reason)}</p></div></div>`;
  }
  return `${templateCard(g.template.name || 'Utility draft', g.template)}
    ${probabilityRow(g.score, null)}
    ${g.findings?.length ? `<div class="notice warn">${icon('triangle-alert')} The checklist flagged promotional wording in this draft: ${g.findings.map(f => esc(f.evidence)).join(', ')}. Edit before submitting.</div>` : ''}
    <p class="muted small">${esc(g.reason || '')} ${esc(g.limitation || '')}</p>
    <div class="actions"><button class="button" id="compose-send-to-check">${icon('scan-text')}Check this draft</button></div>`;
}

function readComposeFields() {
  ['header', 'body', 'footer', 'buttons'].forEach(key => {
    const element = $(`#compose-${key}`);
    if (element) composeState.draft[key] = element.value;
  });
  composeState.draft.requested_category = $('#compose-requested-category')?.value || 'UNKNOWN';
  composeState.draft.meta_category = $('#compose-meta-category')?.value || 'UNKNOWN';
}

function wireCompose() {
  ['header', 'body', 'footer', 'buttons'].forEach(key => {
    const field = $(`#compose-${key}`);
    if (field) field.oninput = () => {
      composeState.draft[key] = field.value;
      // The saved Meta ruling belongs to the uploaded wording, not a new edit.
      composeState.draft.meta_category = 'UNKNOWN';
      const category = $('#compose-meta-category');
      if (category) category.value = 'UNKNOWN';
    };
  });
  ['requested', 'meta'].forEach(key => {
    const field = $(`#compose-${key}-category`);
    if (field) field.onchange = () => { composeState.draft[`${key}_category`] = field.value; };
  });
  const relationship = $('#compose-relationship');
  if (relationship) relationship.onchange = () => { composeState.relationship = relationship.checked; };
  ['task', 'context', 'purpose'].forEach(key => {
    const field = $(`#compose-${key}`);
    if (field) field.oninput = () => { composeState[key] = field.value; };
  });
  document.querySelectorAll('[data-mode]').forEach(button => {
    button.disabled = composeState.busy;
    button.onclick = () => { composeState.mode = button.dataset.mode; renderComposePanel(); };
  });

  const upload = $('#compose-upload'), file = $('#compose-file');
  if (upload) upload.onclick = () => file.click();
  if (file) file.onchange = async () => {
    if (!file.files.length) return;
    const form = new FormData();
    form.append('file', file.files[0]);
    try {
      const extracted = await api('/api/extract', { method: 'POST', body: form });
      composeState.draft = { ...composeState.draft, ...extracted.template,
        requested_category: extracted.requested_category,
        meta_category: extracted.meta_category, format: extracted.format };
      composeState.relationship = false;
      composeState.source = `${file.files[0].name} — first of ${num(extracted.records_in_file)} record(s)`;
      composeState.result = null;
      renderComposePanel();
      toast('Template loaded. Check it to see the verdict.');
    } catch (error) { toast(error.message, true); }
    file.value = '';
  };

  const check = $('#compose-check');
  if (check) check.onclick = async () => {
    readComposeFields();
    composeState.relationship = $('#compose-relationship').checked;
    if (!composeState.draft.body.trim()) { toast('Enter a message body.', true); return; }
    composeState.result = null;
    composeState.busy = true; renderComposePanel();
    try {
      composeState.result = await post('/api/convert', { ...composeState.draft, purpose: composeState.purpose || 'unknown', relationship_confirmed: composeState.relationship });
    } catch (error) { toast(error.message, true); }
    composeState.busy = false; renderComposePanel();
  };

  const accept = $('#compose-accept');
  if (accept) accept.onclick = () => {
    composeState.draft = { ...composeState.draft, ...composeState.result.utility,
      requested_category: 'UTILITY', meta_category: 'UNKNOWN' };
    composeState.result = null;
    renderComposePanel();
    toast('Utility version applied. Check it again to confirm the score.');
  };

  const keep = $('#compose-keep');
  if (keep) keep.onclick = () => { composeState.result = null; renderComposePanel(); toast('Kept as it is.'); };

  const generate = $('#compose-generate');
  if (generate) generate.onclick = async () => {
    composeState.task = $('#compose-task').value;
    composeState.context = $('#compose-context').value;
    composeState.purpose = $('#compose-purpose').value;
    if (!composeState.task.trim()) { toast('Describe the message you need.', true); return; }
    composeState.generated = null;
    composeState.busy = true; renderComposePanel();
    try {
      composeState.generated = await post('/api/generate', { task: composeState.task, purpose: composeState.purpose, context: composeState.context });
    } catch (error) { toast(error.message, true); }
    composeState.busy = false; renderComposePanel();
  };

  const send = $('#compose-send-to-check');
  if (send) send.onclick = () => {
    const t = composeState.generated.template;
    composeState.draft = { name: t.name || '', header: t.header, body: t.body, footer: t.footer, buttons: t.buttons };
    composeState.mode = 'inspect';
    composeState.relationship = false;
    composeState.result = null;
    renderComposePanel();
  };
}
