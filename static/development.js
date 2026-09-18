let developmentTab = 'progress';
const developmentTabs = { progress: 'Progress', data: 'Data journey', learning: 'How it works', evidence: 'Results', code: 'Code map' };

function developmentSection(title, body) {
  return `<section class="development-section"><h2>${title}</h2>${body}</section>`;
}

function developmentContent() {
  const s = state.summary;
  const r = state.model.report;
  if (developmentTab === 'progress') return `
    <div class="development-columns"><section class="development-section"><h2>Built and verified</h2>
    <ol class="milestones">${[
      ['Data foundation', 'Imported the supplied export into SQLite. Requested categories and recorded categories remain separate.'],
      ['Label provenance confirmed', 'On 18 September 2026, the platform owner confirmed that metaTemplateCategory is Meta\'s final category and VERIFIED means successful Meta verification.'],
      ['Dataset workspace', 'Search, filters, record inspection, pagination, import mapping and CSV export.'],
      ['First predictive model', 'TF-IDF plus logistic regression, evaluated on unseen normalized template families.'],
      ['Template reviewer', 'An independent English checklist, factual utility draft patterns and limited promotional-sentence removal.'],
      ['Error auditing', 'Holdout-only predictions are saved before the serving model is trained on all eligible families.'],
      ['Error-review workspace', 'Full components, separate human annotations, candidate edits, manual Meta outcomes and snapshot-scoped export.'],
      ['Chronological experiments', 'Word, character and latent-semantic models compared on later templates, with lexical near-duplicate screening.'],
    ].map(([title, description]) => `<li>${icon('circle-check')}<div><h3>${title}</h3><p>${description}</p></div><span class="badge utility">Complete</span></li>`).join('')}</ol></section>
    <section class="development-section"><h2>Next milestones</h2><ol class="milestones next">${[
      ['Review prediction errors', 'Check false-utility examples against the actual recipient relationship and business event.', 'Next'],
      ['Decision-time validation', 'Obtain actual Meta decision timestamps and a new untouched evaluation set.', 'Needs data'],
      ['Neural model comparison', 'Compare pretrained sentence embeddings or an LLM; local LSA is already measured.', 'Planned'],
      ['Validated rewriting', 'Collect factual recipient context and observed Meta decisions before expanding beyond limited edits.', 'Needs review'],
    ].map(([title, description, status]) => `<li>${icon('circle')}<div><h3>${title}</h3><p>${description}</p></div><span class="badge">${status}</span></li>`).join('')}</ol></section></div>
    ${developmentSection('Release evidence', '<div class="evidence-strip"><div><strong>40 passed</strong><p>Automated tests at the last backend checkpoint</p></div><div><strong>Desktop + mobile</strong><p>App, error reviews and experiments verified in Chrome</p></div><div><strong>Local only</strong><p>No LLM service or Meta API connected</p></div></div><p class="development-caption">Milestones are maintained development records, not a live CI feed. Checkpoint: 18 September 2026.</p>')}`;
  if (developmentTab === 'data') return `
    ${developmentSection('From export to training examples', `<ol class="data-journey">
      <li><strong>${num(s.total)}</strong><h3>Imported records</h3><p>All formats and workflow statuses remain available for browsing.</p></li>
      <li><strong>${num(s.families)}</strong><h3>Body families</h3><p>Normalized bodies group repeated text, numbers, URLs and placeholders.</p></li>
      <li><strong>${num(s.conflicting_families)}</strong><h3>Conflicting families</h3><p>Eligible families with both labels are excluded from training.</p></li>
      <li><strong>${num(s.training_families)}</strong><h3>Training families</h3><p>One representative per eligible, non-conflicting family.</p></li></ol><p class="development-caption">These are distinct counts, not sequential subtraction. Status, label, format and body-length filters also apply.</p>`)}
    ${developmentSection('Two categories, two different meanings', `<div class="development-columns"><div><h3>Requested category</h3><code>messageBody.templateCategory</code><p>The category selected when the draft was prepared. It is not a training label.</p></div><div><h3>Recorded category</h3><code>metaTemplateCategory</code><p>Meta's final category after verification. The platform owner confirmed this on 18 September 2026, along with VERIFIED meaning successful verification at Meta. No independent Meta API check was performed.</p></div></div><div class="notice warning">${num(s.category_mismatches)} records request utility but record marketing. This snapshot does not establish when or why a category changed.</div>`)}
    ${developmentSection('Training eligibility', '<p>Only VERIFIED or APPROVED records with a MARKETING or UTILITY label, TEXT format and a body of at least 10 characters enter the candidate set. Authentication, rich formats, missing labels and unfinished workflows are excluded. Conflicting families are removed; the latest representative of each clean family is kept.</p><p>Header, body, footer and button text are combined for prediction. Grouping uses the body, so paraphrased duplicates may still cross the split.</p>')}
    ${developmentSection('Privacy boundary', '<p>The raw attachment is not copied into tracked source code. SQLite, model artifacts and error exports stay in Git-ignored local storage. Account and author fields are omitted, but message text can still contain personal information. This app is not authenticated and should remain on localhost.</p>')}`;
  if (developmentTab === 'learning') return `
    ${developmentSection('Three separate decisions', '<div class="evidence-strip"><div><strong>Historical prediction</strong><p>What category does this text resemble in our dataset?</p></div><div><strong>Policy checklist</strong><p>Does the draft contain known promotional signals and sufficient service context?</p></div><div><strong>Meta decision</strong><p>Made externally by Meta. Our app neither submits nor approves templates.</p></div></div>')}
    ${developmentSection('Inside the baseline', `<ol class="milestones">${[
      ['1. Normalize text', 'Replace variable details such as numbers and URLs so the model focuses more on reusable wording.'],
      ['2. Convert words into numbers', 'TF-IDF assigns weights to individual words and pairs of words. Frequent generic terms get less emphasis than distinctive terms. No neural embedding model is involved.'],
      ['3. Learn category weights', 'Logistic regression learns which weighted text features are associated with each recorded category. Balanced class weights reduce the influence of the larger class.'],
      ['4. Test on held-out families', 'A seeded group split holds out 25% of families. The vocabulary and classifier are fitted only on the training portion for this evaluation.'],
      ['5. Fit the serving model', 'After measuring performance and saving holdout predictions, a separate model is fitted on all eligible families for interactive review.'],
    ].map(([title, description]) => `<li><div><h3>${title}</h3><p>${description}</p></div></li>`).join('')}</ol>`)}
    ${developmentSection('A grounded edit, not a category trick', '<div class="development-columns"><div><h3>Illustrative draft</h3><div class="message">Your invoice {{1}} is due on {{2}}. Upgrade today.</div></div><div><h3>Possible limited edit</h3><div class="message">Your invoice {{1}} is due on {{2}}.</div></div></div><p>This example requires a real invoice and a confirmed recipient relationship. The checklist can remove a separate promotional sentence; it cannot invent a transaction or turn an unrelated sales campaign into a service event. Mixed promotional and transactional wording in the same sentence may require manual review.</p>')}
    ${developmentSection('What the model cannot explain', '<p>Learned correlations and similar historical templates are not Meta\'s internal reasoning. The English checklist is explicitly coded, not an LLM. Neither component can inspect images, verify a real customer relationship, guarantee approval or provide calibrated approval probabilities.</p>')}`;
  if (developmentTab === 'evidence') {
    if (!r) return developmentSection('No evaluation available', '<p>A trained baseline is needed before results can be inspected.</p><a class="button" href="#model">Open Model lab</a>');
    const matrix = r.confusion_matrix;
    return `${state.model.stale ? '<div class="notice warning">The dataset has changed. These metrics describe the previous training snapshot, not the current dataset.</div>' : ''}
    ${developmentSection('What happened on unseen families', `<p>${num(r.train_families)} families were used to train the evaluation model; ${num(r.test_families)} were held out. Normalized family overlap: ${num(r.group_overlap)}.</p><div class="table-wrap"><table class="matrix"><caption>Rows: recorded category. Columns: held-out prediction.</caption><thead><tr><th>Recorded / predicted</th><th>Marketing</th><th>Utility</th></tr></thead><tbody><tr><th>Marketing</th><td class="correct">${num(matrix[0][0])}</td><td class="incorrect">${num(matrix[0][1])}</td></tr><tr><th>Utility</th><td class="incorrect">${num(matrix[1][0])}</td><td class="correct">${num(matrix[1][1])}</td></tr></tbody></table></div>`)}
    ${developmentSection('Reading the numbers', `<div class="metric-explanations"><div><strong>${percent(r.accuracy)}</strong><h3>Accuracy</h3><p>Share of all holdout predictions matching the recorded category. Always predicting the training majority class achieves ${percent(r.majority_accuracy)}.</p></div><div><strong>${percent(r.per_class.UTILITY.precision)}</strong><h3>Utility precision</h3><p>Of ${num(r.predicted_utility_count)} utility predictions, ${num(matrix[1][1])} match utility labels. The other ${num(matrix[0][1])} are recorded marketing.</p></div><div><strong>${percent(r.per_class.UTILITY.recall)}</strong><h3>Utility recall</h3><p>Of ${num(r.per_class.UTILITY.support)} recorded utility examples, ${num(matrix[1][1])} were found. ${num(matrix[1][0])} were predicted marketing.</p></div></div><div class="notice warning">This baseline is not reliable enough for automatic utility submissions. A false-utility prediction is especially important to inspect before deployment.</div>`)}
    ${developmentSection('What has actually been tested', '<p>Automated coverage includes parsing, separate label fields, idempotent imports, eligibility filters, conflicting families, grouped evaluation, persisted holdout errors, stale model rejection, policy context, API validation, cross-origin write rejection and CSV formula escaping.</p><p>Browser checks cover search, filters, record inspection, model training, draft edits and desktop/mobile layouts. These checks verify software behavior, not Meta approval.</p>')}
    ${developmentSection('Limits of this evidence', `<p>Label provenance was confirmed by the platform owner on 18 September 2026: metaTemplateCategory is Meta's final category; VERIFIED means successful verification at Meta. No independent Meta API check was performed.</p><ul>${r.limitations.filter(item => !item.includes('upstream semantics need confirmation')).map(item => `<li>${esc(item)}</li>`).join('')}</ul><p class="development-caption">Model snapshot: ${esc(r.trained_at)}. Metrics refresh when this page reloads or Refresh is selected.</p>`)}`;
  }
  return `${developmentSection('Project anatomy', `<div class="code-map">${[
    ['templatelab/data.py', 'Parse exports, preserve labels, normalize families and store records in SQLite.'],
    ['templatelab/model.py', 'Train and evaluate the text classifier, persist artifacts and retrieve similar examples.'],
    ['templatelab/policy.py', 'Apply the limited English checklist and grounded draft/edit patterns.'],
    ['templatelab/audit.py', 'Export false-utility and false-marketing holdout examples.'],
    ['templatelab/error_review.py', 'Persist human annotations and candidate outcomes separately from source labels.'],
    ['templatelab/experiments.py', 'Chronological comparison and lexical overlap screening; serving model is unchanged.'],
    ['templatelab/app.py', 'Expose the local API and serve the browser application.'],
    ['static/', 'Browser views, interactions, styles and bundled icons.'],
    ['tests/', 'Automated backend tests and browser workflow checks.'],
    ['docs/', 'Implementation plan, dataset audit and session handoff.'],
    ['.data/', 'Private local database, model and holdout error report. Excluded from Git.'],
  ].map(([file, description]) => `<div><code>${file}</code><p>${description}</p></div>`).join('')}</div>`)}
  ${developmentSection('Technical choices', '<dl class="technology-list"><dt>FastAPI</dt><dd>Python HTTP API and local web server.</dd><dt>SQLite</dt><dd>Single-file local data storage without a database service.</dd><dt>scikit-learn</dt><dd>Established text vectorization, classification and evaluation implementations.</dd><dt>Vanilla JavaScript</dt><dd>Browser interaction without a frontend build pipeline.</dd><dt>pytest + Playwright</dt><dd>Backend behavior tests and real-browser workflow checks.</dd></dl>')}
  ${developmentSection('Not built yet', '<p>No external LLM calls, neural embeddings, general-purpose rewriting, Meta submission, production authentication, billing integration or verified cost-savings measurement. Your approximate 7.5:1 cost ratio motivated the project; it is not a current pricing quote or an implemented billing calculation.</p>')}`;
}

function renderDevelopment() {
  const r = state.model.report;
  $('#app').innerHTML = heading('Template Lab development', 'Working prototype / research baseline / human review required', `<button class="icon-button" id="refresh-development" aria-label="Refresh development data" title="Refresh development data">${icon('refresh-cw')}</button><a class="button primary" href="#review">${icon('scan-text')}Open reviewer</a>`) +
    `<div class="stats">${stat('Imported records', num(state.summary.total), 'Current local dataset', 'database')}${stat('Eligible families', num(state.summary.training_families), 'After filtering and deduplication', 'layers')}${stat('Utility precision', r ? percent(r.per_class.UTILITY.precision) : 'Not trained', state.model.stale ? 'Previous snapshot; retraining needed' : 'Held-out evaluation', 'target', 'amber')}${stat('Deployment', 'Prototype', 'No automated Meta submissions', 'flask-conical')}</div>
    <div class="development-tabs" role="tablist" aria-label="Development sections">${Object.entries(developmentTabs).map(([key, label]) => `<button role="tab" id="dev-tab-${key}" aria-controls="development-panel" aria-selected="${developmentTab === key}" tabindex="${developmentTab === key ? 0 : -1}" data-development-tab="${key}">${label}</button>`).join('')}</div>
    <div id="development-panel" role="tabpanel" aria-labelledby="dev-tab-${developmentTab}" tabindex="0">${developmentContent()}</div>`;
  document.querySelectorAll('[data-development-tab]').forEach(button => {
    button.onclick = () => { developmentTab = button.dataset.developmentTab; renderDevelopment(); $(`#dev-tab-${developmentTab}`).focus(); };
    button.onkeydown = event => {
      const keys = Object.keys(developmentTabs);
      let index = keys.indexOf(developmentTab);
      if (event.key === 'ArrowRight') index = (index + 1) % keys.length;
      else if (event.key === 'ArrowLeft') index = (index + keys.length - 1) % keys.length;
      else if (event.key === 'Home') index = 0;
      else if (event.key === 'End') index = keys.length - 1;
      else return;
      event.preventDefault(); developmentTab = keys[index]; renderDevelopment(); $(`#dev-tab-${developmentTab}`).focus();
    };
  });
  $('#refresh-development').onclick = async event => {
    event.currentTarget.disabled = true;
    try { await refresh(); if (state.route === 'development') renderDevelopment(); }
    catch (error) { toast(error.message, true); if ($('#refresh-development')) $('#refresh-development').disabled = false; }
  };
  icons();
}
