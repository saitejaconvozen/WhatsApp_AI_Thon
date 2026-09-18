const templateForm = document.querySelector('#template-form');
const taskForm = document.querySelector('#task-form');
const result = document.querySelector('#result');
const draft = document.querySelector('#draft');
const submitTemplate = document.querySelector('#submit-template');
const submitTask = document.querySelector('#submit-task');
const templateWorkspace = document.querySelector('#template-workspace');
const taskWorkspace = document.querySelector('#task-workspace');
const relationshipField = document.querySelector('#relationship-field');
const inputHeading = document.querySelector('#input-heading');
const outputHeading = document.querySelector('#output-heading');

const MODES = {
  classify: { label: 'Classify template', icon: 'scan-text', output: 'Predicted Meta category', input: 'Template' },
  convert: { label: 'Convert to utility', icon: 'wand-sparkles', output: 'Conversion', input: 'Template to convert' },
};
let mode = 'classify';

const element = (tag, properties = {}) => Object.assign(document.createElement(tag), properties);
const paragraph = (text, className = 'muted') => element('p', { className, textContent: text });
const percent = value => value == null ? 'Not applicable' : `${(value * 100).toFixed(1)}%`;

function icons() { window.lucide?.createIcons(); }
icons();

function definitionList(rows) {
  const list = element('dl');
  for (const [label, value] of rows) {
    const row = element('div');
    row.append(element('dt', { textContent: label }), element('dd', { textContent: value }));
    list.append(row);
  }
  return list;
}

function templateCard(title, template) {
  const card = element('div', { className: 'card' });
  card.append(element('h3', { textContent: title }));
  for (const [key, label] of [['header', 'Header'], ['footer', 'Footer'], ['buttons', 'Buttons']]) {
    if (template[key]) card.append(element('p', { className: 'component', textContent: `${label}: ${template[key]}` }));
  }
  card.append(element('p', { className: 'body-text', textContent: template.body || '' }));
  return card;
}

function looksSpliced(body) {
  const greeting = /\b(?:hi|hello|dear|hey)\b[\s,]/gi;
  const found = [...body.matchAll(greeting)].map(m => m.index);
  // A greeting well past the opening usually means a second template was pasted
  // on top of the first. That once made a correct verdict look wrong.
  return found.some(index => index > 60);
}

function checkSplice() {
  const warning = document.querySelector('#splice-warning');
  if (!warning) return;
  const body = templateForm.body.value;
  const spliced = looksSpliced(body);
  warning.hidden = !spliced;
  if (spliced) warning.textContent = 'This looks like two templates pasted together — there is a greeting partway through the body. Classify one template at a time, or the mixed content will be judged as one message.';
}

function setMode(next) {
  mode = next;
  document.querySelectorAll('[data-mode]').forEach(tab => tab.setAttribute('aria-selected', String(tab.dataset.mode === next)));
  const onTask = next === 'generate';
  taskWorkspace.hidden = !onTask;
  templateWorkspace.hidden = onTask;
  if (!onTask) {
    const config = MODES[next];
    inputHeading.textContent = config.input;
    outputHeading.textContent = config.output;
    submitTemplate.querySelector('span').textContent = config.label;
    // Lucide replaces the <i data-lucide> placeholder with an <svg>, so after the
    // first paint the icon has to be re-created from a fresh placeholder.
    const glyph = submitTemplate.querySelector('i, svg');
    if (glyph) {
      // `dataset` is a read-only accessor, so it has to be set as an attribute.
      const placeholder = document.createElement('i');
      placeholder.setAttribute('data-lucide', config.icon);
      glyph.replaceWith(placeholder);
    }
    relationshipField.hidden = next !== 'convert';
    result.replaceChildren(paragraph('Nothing checked yet.'));
    icons();
  }
}

document.querySelectorAll('[data-mode]').forEach(tab => { tab.onclick = () => setMode(tab.dataset.mode); });
templateForm.onreset = () => {
  result.replaceChildren(paragraph('Nothing checked yet.'));
  const warning = document.querySelector('#splice-warning');
  if (warning) warning.hidden = true;
};
templateForm.body.addEventListener('input', checkSplice);
taskForm.onreset = () => draft.replaceChildren(paragraph('No draft yet.'));

async function send(url, payload, target, button, busyLabel, render) {
  const original = button.querySelector('span').textContent;
  button.disabled = true;
  button.querySelector('span').textContent = busyLabel;
  target.replaceChildren(paragraph('Working...'));
  try {
    const response = await fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Check the fields and try again.');
    target.replaceChildren(...render(data));
  } catch (error) {
    target.replaceChildren(element('p', { className: 'error', textContent: error.message }));
  } finally {
    button.disabled = false;
    button.querySelector('span').textContent = original;
  }
}

function renderPrediction(data) {
  const category = element('p', { className: `category ${data.category === 'UTILITY' ? 'UTILITY' : 'MARKETING'}`, textContent: data.category });
  return [category, definitionList([
    [data.score_label || 'Utility score', percent(data.utility_probability)],
    ['Review band', data.band || 'Not available'],
    ['Model', data.model],
  ]), paragraph(data.notice, 'notice'),
  element('div', { id: 'why', className: 'why' })];
}

function renderExplanation(data) {
  const box = document.querySelector('#why');
  if (!box) return;
  if (!data.available) {
    box.replaceChildren(element('h3', { textContent: 'Why' }), paragraph(data.reason || data.notice));
    return;
  }
  const nodes = [element('h3', { textContent: 'Why' })];
  if (data.agrees === false) {
    nodes.push(element('p', {
      className: 'disagree',
      textContent: `The explaining model reads this as ${data.category}, the scoring model as ${data.local_category}. They disagree about as often as they are each right, so treat this one as needing a human.`,
    }));
  }
  if (data.rationale) nodes.push(element('p', { className: 'rationale', textContent: data.rationale }));
  if (data.clauses?.length) {
    const list = element('ul', { className: 'clauses' });
    data.clauses.forEach(c => {
      const item = element('li');
      item.append(element('strong', { textContent: c.id }), document.createTextNode(c.text ? ` — ${c.text}` : ''));
      list.append(item);
    });
    nodes.push(element('p', { className: 'component', textContent: 'Policy clauses cited' }), list);
  }
  nodes.push(paragraph(data.notice, 'notice'));
  box.replaceChildren(...nodes);
}

async function explain(payload) {
  const box = document.querySelector('#why');
  if (box) box.replaceChildren(element('h3', { textContent: 'Why' }), paragraph('Asking the model for its reasoning...'));
  try {
    const response = await fetch('/api/explain', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
    renderExplanation(await response.json());
  } catch (error) {
    renderExplanation({ available: false, reason: error.message });
  }
}

function renderConversion(data) {
  const nodes = [element('p', { className: 'category', textContent: (data.verdict || '').replace(/_/g, ' ') })];
  if (data.reason) nodes.push(paragraph(data.reason, 'notice'));
  const before = data.before?.available ? percent(data.before.utility_probability) : 'Not available';
  const after = data.after?.available ? percent(data.after.utility_probability) : null;
  nodes.push(definitionList([['Utility score now', before], ...(after ? [['After the change', after]] : []), ['Method', data.method || '—']]));
  if (data.utility) nodes.push(templateCard('Utility version', data.utility));
  if (data.split_off) nodes.push(templateCard('Promotional part, send separately as marketing', { body: data.split_off.body }));
  if (data.removed?.length) {
    const list = element('ul', { className: 'removed' });
    data.removed.forEach(r => list.append(element('li', { textContent: `${r.component}: ${r.text}` })));
    nodes.push(element('h3', { textContent: 'Removed' }), list);
  }
  if (data.needs_human && data.ambiguous?.length) {
    const list = element('ul', { className: 'removed' });
    data.ambiguous.forEach(a => list.append(element('li', { textContent: a.text })));
    nodes.push(element('h3', { textContent: 'Kept for human review' }), list);
  }
  nodes.push(paragraph(data.notice, 'notice'));
  return nodes;
}

function renderDraft(data) {
  if (!data.possible) {
    return [element('p', { className: 'category MARKETING', textContent: (data.verdict || 'NOT POSSIBLE').replace(/_/g, ' ') }),
            paragraph(data.reason || '', 'notice')];
  }
  const nodes = [templateCard(data.template.name || 'Utility draft', data.template),
                 definitionList([['Service event', data.purpose || 'unknown'],
                                 ['Utility score', percent(data.score?.utility_probability)],
                                 ['Method', data.method || '—']])];
  if (data.findings?.length) {
    nodes.push(element('p', { className: 'error', textContent: `Promotional wording flagged: ${data.findings.map(f => f.code).join(', ')}. Edit before submitting.` }));
  }
  if (data.reason) nodes.push(paragraph(data.reason, 'notice'));
  nodes.push(paragraph(data.notice, 'notice'));
  return nodes;
}

templateForm.onsubmit = event => {
  event.preventDefault();
  const payload = Object.fromEntries(new FormData(templateForm));
  payload.relationship_confirmed = templateForm.relationship_confirmed?.checked ?? false;
  if (mode === 'classify') {
    // The explanation is a second, slower call so the verdict is never held up by it.
    return send('/api/predict', payload, result, submitTemplate, 'Classifying...', renderPrediction)
      .then(() => explain(payload));
  }
  return send('/api/convert', payload, result, submitTemplate, 'Converting...', renderConversion);
};

taskForm.onsubmit = event => {
  event.preventDefault();
  return send('/api/generate', Object.fromEntries(new FormData(taskForm)), draft, submitTask, 'Drafting...', renderDraft);
};
