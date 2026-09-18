const $ = selector => document.querySelector(selector);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const count = value => Number(value || 0).toLocaleString();
const pct = value => `${(100 * Number(value || 0)).toFixed(1)}%`;
const icon = name => `<i data-lucide="${name}"></i>`;
const paintIcons = () => window.lucide?.createIcons();
let report, errorPage = 1, direction = 'all', reviewStatus = 'all', query = '', subset = 'all';
const metric = (label, value, note) => `<div class="metric"><span>${label}</span><strong>${value}</strong><small>${note}</small></div>`;
const badge = category => `<span class="pill ${category === 'UTILITY' ? 'green' : category === 'MARKETING' ? 'amber' : 'gray'}">${escapeHtml(category)}</span>`;
const sectionHead = (title, text, tag = '') => `<div class="section-head"><div><h2>${title}</h2><p>${text}</p></div>${tag ? `<span class="pill">${tag}</span>` : ''}</div>`;
const bar = (label, value, tone = '') => `<div><div class="bar-label"><span>${escapeHtml(label)}</span><strong>${pct(value)}</strong></div><div class="bar-track" role="img" aria-label="${escapeHtml(label)}: ${pct(value)}"><div class="bar-fill ${tone}" style="width:${Math.min(100, Math.max(0, value * 100))}%"></div></div></div>`;

async function loadResults() {
  $('#refresh').disabled = true;
  $('#load-status').textContent = 'Reading local results...';
  $('#load-status').classList.remove('failure');
  try {
    const response = await fetch(window.RESULTS_ENDPOINT || '/api/results');
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `Results unavailable (${response.status}).`);
    }
    report = await response.json();
    $('#load-status').textContent = `Loaded ${new Date(report.loaded_at).toLocaleString()} | Read-only view | Milestones are maintained checkpoints, not a live CI feed.`;
    $('#download').disabled = false;
    errorPage = 1;
    render();
  } catch (error) {
    $('#load-status').textContent = `${error.message} ${report ? 'Previously loaded results remain below.' : 'Use Refresh to retry.'}`;
    $('#load-status').classList.add('failure');
    if (!report) $('#results').innerHTML = '<div class="empty">No results could be loaded.</div>';
  } finally { $('#refresh').disabled = false; }
}

function render() {
  const s = report.dataset, b = report.baseline, r = b.report, e = report.experiments;
  const distribution = [['MARKETING','#d3a051'],['UTILITY','#509e86'],['AUTHENTICATION','#7297c4'],['UNKNOWN','#c5cdca']];
  $('#results').innerHTML = `
    <section id="overview">${sectionHead('The project at a glance', 'Meta-assigned categories; verified templates; local-only analysis.', 'Research prototype')}
      <div class="metrics">${metric('Imported templates', count(s.total), 'All formats and workflow statuses')}${metric('Eligible families', count(s.training_families), 'Distinct, non-conflicting text examples')}${metric('Baseline utility precision', r ? pct(r.per_class.UTILITY.precision) : 'Not trained', 'Original random holdout')}${metric('Human-reviewed errors', report.errors.counts ? `${count(report.errors.counts.reviewed)} / ${count(report.errors.counts.total)}` : 'Unavailable', 'Original baseline error queue')}</div>
      <div class="notice">No model is approved for automatic utility submissions. The serving baseline remains unchanged by the comparison experiments.</div>
      <div class="grid-two"><div><h3>Recorded Meta categories</h3><div class="distribution" role="img" aria-label="${escapeHtml(distribution.map(([key]) => `${key}: ${count(s.categories[key])}`).join(', '))}">${distribution.map(([key,color]) => `<span style="width:${s.total ? (s.categories[key] || 0) / s.total * 100 : 0}%;background:${color}"></span>`).join('')}</div><div class="legend">${distribution.map(([key,color]) => `<span><i class="swatch" style="background:${color}"></i>${key === 'UNKNOWN' ? 'Unrecorded' : key.toLowerCase()} <strong>${count(s.categories[key])}</strong></span>`).join('')}</div></div>
      <div><h3>Data quality</h3><dl class="facts"><div><dt>Requested utility / recorded marketing</dt><dd>${count(s.category_mismatches)}</dd></div><div><dt>Conflicting training families excluded</dt><dd>${count(s.conflicting_families)}</dd></div><div><dt>Repeated records beyond first family member</dt><dd>${count(s.duplicate_records)}</dd></div></dl></div></div>
    </section>
    <section id="baseline">${sectionHead('Original baseline', 'Word TF-IDF + logistic regression. Evaluation model trained separately from the serving model.', 'Random grouped holdout')}
      ${r ? baselineHtml(r, b.stale) : '<div class="empty">No baseline has been trained. Train it in the main workspace.</div>'}
    </section>
    <section id="comparison">${sectionHead('Model comparison', 'Older template families used for training; later families used for evaluation.', 'Chronological snapshot')}
      ${e.available ? experimentsHtml(e) : '<div class="empty">No experiment report is available. Run a comparison in the main workspace.</div>'}
    </section>
    <section id="deepseek">${sectionHead('DeepSeek pilot', 'Historical Meta-label prediction with training-only retrieval.', 'Requested-utility sample')}${pilotMarkup(report.benchmark)}</section>
    <section id="training">${sectionHead('Accuracy improvement', 'Fixed training, validation and historical test partitions.', 'Research in progress')}${improvementMarkup(report.improvement)}</section>
    <section id="errors">${sectionHead('Actual error examples', 'The original random-holdout errors, not the chronological experiment errors.', 'Full template inspection')}
      ${report.errors.available ? `<div class="toolbar"><input id="error-query" type="search" aria-label="Search error examples" placeholder="Search template names or any message component" value="${escapeHtml(query)}"><select id="error-direction" aria-label="Error direction"><option value="all">Both error directions</option><option value="false_utility">False utility</option><option value="false_marketing">False marketing</option></select><select id="review-status" aria-label="Review status"><option value="all">All review statuses</option><option value="pending">Pending</option><option value="context_needed">Context needed</option><option value="reviewed">Reviewed</option></select></div><div id="error-table"></div>` : `<div class="notice">${escapeHtml(report.errors.reason)}</div>`}
    </section>
    <section id="conversion">${sectionHead('Converting marketing to utility', 'Every template the business submitted as utility and Meta recorded as marketing, run through the converter.', 'Batch result')}
      ${report.conversions?.available ? conversionHtml(report.conversions) : `<div class="empty">${escapeHtml(report.conversions?.reason || 'No batch conversion has been run.')}</div>`}
      <div id="conversion-outputs"></div>
    </section>
    <section id="ideas">${sectionHead('What could be built next', 'Every direction identified for this project, with what has already been tried and what it would take.', 'Roadmap')}
      ${ideasHtml()}
    </section>
    <section id="progress">${sectionHead('Development progress', 'Completed software and research work, with outstanding validation kept separate.', 'Checkpoint: 18 Sep 2026')}
      <div class="grid-two"><div><h3>Implemented</h3><ul class="timeline">${[
        ['Dataset foundation','Import, label separation, eligibility filters, family normalization and local SQLite storage.'],
        ['Model and template review','Random-holdout baseline, similar examples, limited English checklist and grounded draft patterns.'],
        ['Error-review workflow','Snapshot-scoped notes, recipient context, candidate edits and manual Meta outcomes.'],
        ['Stronger evaluation','Chronological comparison of word, character and local LSA models, with lexical overlap screening.'],
        ['Unified results site','Read-only data, evaluation, error inspection and development evidence in one place.'],
        ['Calibrated decisions','Probabilities, a precision-targeted abstention band, and a portfolio risk budget priced in money rather than accuracy.'],
        ['Authoring workflow','Check or upload a template, convert it toward utility while keeping the transactional anchor, split off the promotional half, or draft a new utility template from a task.'],
      ].map(([title,text]) => `<li>${icon('circle-check')}<div><strong>${title}</strong><p>${text}</p></div></li>`).join('')}</ul></div>
      <div><h3>Still required</h3><ul class="timeline future">${[
        ['Human error analysis','Review the actual templates against business events and recipient relationships.'],
        ['Historical decision evidence','Obtain Meta decision timestamps and revisions; current dates are only proxies.'],
        ['Improve semantic prediction','The DeepSeek pilot and pretrained embedding comparison are research evidence, not a 90% solution. Compare adapted or hybrid models on separate validation data.'],
        ['Validated rewriting','Conversion and drafting exist; neither has been checked against an actual Meta decision yet.'],
        ['Ground truth from Meta','Harvest correct_category from the Message Templates API. The one input that raises the ceiling rather than working beneath it.'],
        ['Deployment decision','Use a new untouched evaluation set. No automatic model promotion or Meta submission is enabled.'],
      ].map(([title,text]) => `<li>${icon('circle')}<div><strong>${title}</strong><p>${text}</p></div></li>`).join('')}</ul></div></div>
      <details><summary>Architecture, label provenance and privacy</summary><ul><li>FastAPI serves the local API, SQLite stores records, scikit-learn trains models, and vanilla JavaScript renders the websites.</li><li>The platform owner confirmed that metaTemplateCategory is Meta's final category and VERIFIED means successful Meta verification. No independent Meta API check was performed.</li><li>The checklist does not reveal Meta's internal reasoning. Model predictions are not approval probabilities.</li><li>Hosted LLM calls are opt-in; no Meta submission API is connected. ${report.published ? 'This published build carries aggregate metrics only; template text is never included in it.' : 'Reports contain template text and may contain personal information; keep this site on localhost.'}</li><li>Development milestones are manual checkpoints. Data counts, model reports and error annotations are loaded from the shared local workspace when refreshed.</li></ul></details>
    </section>`;
  if (e.available) {
    $('#comparison-subset').value = subset;
    $('#comparison-subset').onchange = event => { subset = event.target.value; renderComparison(); };
    renderComparison();
  }
  loadOutputs();
  if (report.errors.available) {
    $('#error-direction').value = direction;
    $('#review-status').value = reviewStatus;
    $('#error-query').oninput = event => { query = event.target.value; errorPage = 1; renderErrors(); };
    $('#error-direction').onchange = event => { direction = event.target.value; errorPage = 1; renderErrors(); };
    $('#review-status').onchange = event => { reviewStatus = event.target.value; errorPage = 1; renderErrors(); };
    renderErrors();
  }
  paintIcons();
}

function baselineHtml(r, stale) {
  const m = r.confusion_matrix;
  return `${stale ? '<div class="notice">Dataset changed since this model was trained. These metrics describe an older snapshot.</div>' : ''}
    <div class="grid-two"><div><div class="table-wrap"><table class="matrix"><caption>Actual recorded category by held-out prediction</caption><thead><tr><th>Recorded / predicted</th><th>Marketing</th><th>Utility</th></tr></thead><tbody><tr><th>Marketing</th><td class="correct">${count(m[0][0])}</td><td class="wrong">${count(m[0][1])}</td></tr><tr><th>Utility</th><td class="wrong">${count(m[1][0])}</td><td class="correct">${count(m[1][1])}</td></tr></tbody></table></div></div>
    <div><h3>Measured performance</h3><div class="bars">${bar('Accuracy', r.accuracy)}${bar('Utility precision', r.per_class.UTILITY.precision, 'green')}${bar('Utility recall', r.per_class.UTILITY.recall, 'amber')}</div><p class="small" style="margin-top:18px">${count(r.train_families)} training families / ${count(r.test_families)} holdout families. Majority accuracy: ${pct(r.majority_accuracy)}. Normalized family overlap: ${count(r.group_overlap)}.</p></div></div>
    <div class="notice blue">Of ${count(r.predicted_utility_count)} utility predictions, ${count(m[1][1])} matched Meta's utility label and ${count(m[0][1])} were recorded marketing. That is what utility precision measures.</div>
    ${r.slices ? slicesHtml(r.slices) : ''}
    ${report.reliability ? reliabilityHtml(report.reliability) : ''}
    ${r.bands ? bandHtml(r.bands, r.thresholds) : ''}
    <details><summary>Baseline methodology and limitations</summary><p class="small">Snapshot: ${escapeHtml(r.trained_at)}. ${escapeHtml(r.split)}</p><ul>${r.limitations.map(text => `<li>${escapeHtml(text)}</li>`).join('')}</ul></details>`;
}

const STATUS = {
  shipped: ['Shipped', 'green', 'circle-check'],
  ready: ['Built, not enabled', 'amber', 'circle-dashed'],
  measured: ['Tried, did not work', 'gray', 'circle-slash'],
  proposed: ['Proposed', 'blue', 'circle'],
  blocked: ['Needs data we do not have', 'gray', 'circle-help'],
};

const IDEAS = [
  ['Data coverage', 'Roughly a quarter of the export is never modelled, and some signal is discarded at parse time.', [
    ['ready', 'Sentence embeddings', 'A local pretrained encoder as a fifth comparison arm. Wired into Experiments and currently recorded as skipped; running it needs the encoder installed.'],
    ['proposed', 'Model MEDIA templates', '790 records, 19% of the export, are excluded because only TEXT is eligible. Header type and caption text are already parsed and could be modelled without touching the image itself.'],
    ['proposed', 'Model CAROUSEL and the long tail', '205 carousels plus 46 reply, 45 call-permission, 9 list and 1 limited-time-offer records. Each needs its own component handling.'],
    ['proposed', 'Capture button type, not just label', 'Quick-reply and CTA/URL buttons are flattened to their text at parse time, so a link button and a reply button look identical. CTA buttons are a standard marketing marker. Needs a re-import migration because record ids embed the family hash.'],
    ['proposed', 'Capture CTA link targets', 'A button pointing at an order page and one pointing at a storefront are opposite signals and currently indistinguishable.'],
    ['proposed', 'Use unlabelled records', '351 records have no recorded category and 458 are pending, failed or rejected. They cannot supervise a classifier but can pretrain a representation.'],
    ['proposed', 'Real language detection', 'Every record claims ENGLISH_US and the only check is a non-ASCII character ratio. Mixed-language bodies are scored as if English.'],
    ['blocked', 'Meta decision timestamps', 'The export carries creation and update dates, not when Meta decided. Without them the chronological split is a proxy, not a decision-time test.'],
    ['blocked', 'Template revision history', 'Only current text and current label are stored, so a template edited after a rejection is indistinguishable from one approved first time.'],
    ['blocked', 'Recipient relationship metadata', 'The utility verdict and the rewriter both hinge on a checkbox nobody verifies. Real trigger and relationship data would replace self-attestation.'],
  ]],
  ['Modelling', 'The serving model is one linear classifier over word n-grams. These are the alternatives worth measuring against it.', [
    ['shipped', 'Calibrated probabilities and an abstention band', 'Predictions carry a calibrated probability and refer uncertain templates instead of guessing.'],
    ['ready', 'LLM reviewer', 'Judges a draft against versioned category definitions with retrieved precedents, and explains itself. Off until a backend and an egress acknowledgement are both set.'],
    ['measured', 'Structural feature block', 'Placeholder, button, emoji and punctuation counts. Made every metric worse on two independent splits; TF-IDF already encodes them. Kept as an experiment arm.'],
    ['measured', 'Requested category as a feature', 'Near-deterministic and would leak: a requested-marketing template is always recorded marketing. Used as a gate instead, never as an input.'],
    ['shipped', 'Cost-sensitive decision layer', 'A cost model, a per-template break-even and a portfolio risk budget. Validated on the real holdout: actual misclassification stayed inside the budget at every level tested.'],
    ['proposed', 'Conformal prediction', 'Would give the abstention band a statistical coverage guarantee instead of a precision target tuned on out-of-fold predictions.'],
    ['proposed', 'Per-component models', 'Header, body, footer and buttons are concatenated into one string. The checklist already treats them separately and finds promotional content in buttons specifically.'],
    ['proposed', 'Ensemble the three readings', 'Checklist, model and LLM are deliberately unreconciled. A combiner could be measured — but only against the risk of hiding disagreement that a reviewer should see.'],
    ['proposed', 'Train only on the live decision', 'Fit exclusively on requested-utility families, matching the condition the model is actually used in, rather than learning from cases that were never in doubt.'],
    ['proposed', 'Interpretable explanations', 'The classifier is linear, so the n-grams driving each prediction can be surfaced directly. Cheaper and more faithful than asking a model to narrate itself.'],
  ]],
  ['Evaluation', 'The current numbers come from one split of 1,923 families and are reported without error bars.', [
    ['shipped', 'Report by what was requested', 'Separates the live decision from cases that were never in doubt.'],
    ['proposed', 'Confidence intervals', 'Every figure on this page is a point estimate. On 365 families the requested-utility accuracy carries several points of sampling error either way.'],
    ['proposed', 'Nested cross-validation', 'A single 25% holdout on a dataset this size makes model comparison noisy. Repeated splits would separate real differences from split luck.'],
    ['proposed', 'An untouched final test set', 'Set aside before any model selection and opened once. Nothing here has been validated that way, so every comparison has seen the same data.'],
    ['proposed', 'Semantic duplicate detection', 'The overlap audit is lexical character n-grams at 0.90 cosine. A reworded template with the same purpose passes straight through it.'],
    ['proposed', 'Investigate the chronological gap', 'Accuracy drops from 79.8% on the random holdout to about 70% on the date-based split. That is either drift in Meta behaviour, drift in what was submitted, or an artefact of the cutoff — currently unresolved.'],
    ['proposed', 'Slice errors by sender', 'Template names carry brand prefixes. If errors concentrate in a few senders, that is a content problem rather than a model problem.'],
  ]],
  ['Review workflow', 'The human loop exists but nothing has been reviewed through it yet.', [
    ['proposed', 'Stop discarding annotations on retrain', 'The queue is keyed to a snapshot hash of dataset revision plus training time, so every retrain empties it. Reviews should follow the record, not the snapshot.'],
    ['proposed', 'Rank the queue by uncertainty', 'Templates nearest the decision boundary teach the most per review. The queue is currently unordered.'],
    ['proposed', 'Record real Meta outcomes', 'The schema for observed outcomes exists and is unused. Populating it builds the only genuinely prospective validation set available.'],
    ['shipped', 'Convert toward utility', 'Splits a mixed template, keeps the transactional anchor, and offers the promotional half as a separate marketing template. Refuses rather than inventing a transaction. Re-scores and shows both probabilities.'],
    ['proposed', 'Batch scoring', 'Score a file of drafts and return bands, so the team can triage a backlog instead of checking templates one at a time.'],
    ['proposed', 'Cite policy clauses in the checklist', 'The LLM reviewer cites rule ids from a versioned definitions file. The regex checklist has no provenance for any of its rules.'],
  ]],
  ['Decision and authoring', 'What the tool now does with a prediction, rather than just reporting one.', [
    ['shipped', 'Authoring-time check', 'Paste or upload a template and get the verdict before submitting it to Meta, with the flagged fragments named.'],
    ['shipped', 'Split-template workflow', 'The highest-value automated fix: the transactional part as UTILITY to everyone, the promotional rider as a separate MARKETING template to an opted-in segment.'],
    ['shipped', 'Utility template generation', 'Draft a template from a described task. Refuses promotional tasks instead of manufacturing a transactional wrapper for them.'],
    ['shipped', 'Minimal-pair regression tests', 'Assert behaviour on decisive pairs, not accuracy. Added because structural features raised utility precision to 80.9% while F1 fell from 70.9% to 62.4% \u2014 the headline metric hid it.'],
    ['shipped', 'Nearest-opposite retrieval', 'Few-shot examples take the closest precedent from each category rather than the top-k overall, so the prompt frames the contrast instead of arguing by weight of numbers.'],
    ['ready', 'LLM rewriting and drafting', 'Both the converter and the generator use a model when one is configured, and fall back to the checklist and built-in patterns when not.'],
    ['proposed', 'Batch scoring', 'Score a file of drafts and return bands, so a backlog can be triaged rather than checked one at a time.'],
  ]],
  ['Engineering', 'Mostly debt that will bite later rather than now.', [
    ['shipped', 'Feature version on the model artifact', 'Dataset revision alone could not detect a pipeline change, so a stale model could be served as current.'],
    ['shipped', 'Metrics-only published build', 'An allowlist separates aggregates from template content so results can be shared without the corpus.'],
    ['proposed', 'Decouple record id from family', 'Ids are hashed from a dict that includes the family key, so any grouping change would duplicate the dataset on re-import. The current fix derives grouping at read time and works around it rather than solving it.'],
    ['proposed', 'Continuous integration', 'The suite is run by hand. Nothing stops a regression reaching the working tree.'],
    ['proposed', 'Experiment tracking', 'Each run overwrites the last report, so results are not comparable across time and a regression in a comparison arm would go unnoticed.'],
    ['proposed', 'Authentication', 'Neither app has any. Both are loopback-only for that reason, which is also why they cannot be shared as they stand.'],
  ]],
];

function ideasHtml() {
  const counts = {};
  IDEAS.forEach(([, , items]) => items.forEach(([status]) => { counts[status] = (counts[status] || 0) + 1; }));
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  return `<div class="metrics">${Object.entries(STATUS).map(([key, [label]]) =>
      metric(label, count(counts[key] || 0), key === 'measured' ? 'Kept as a reproducible arm' : '')).join('')}</div>
    <p class="small" style="margin-top:8px">${count(total)} ideas across ${IDEAS.length} areas. Status reflects this repository, not a commitment or a schedule.</p>
    ${IDEAS.map(([group, note, items]) => `<h3 style="margin-top:28px">${escapeHtml(group)}</h3>
      <p class="small">${escapeHtml(note)}</p>
      <ul class="timeline ideas">${items.map(([status, title, text]) => {
        const [label, tone, mark] = STATUS[status];
        return `<li>${icon(mark)}<div><strong>${escapeHtml(title)}</strong> <span class="pill ${tone}">${escapeHtml(label)}</span><p>${escapeHtml(text)}</p></div></li>`;
      }).join('')}</ul>`).join('')}`;
}

function reliabilityHtml(rel) {
  const cv = rel.cross_validation;
  return `<h3 style="margin-top:28px">How much to trust these numbers</h3>
    <div class="table-wrap"><table class="comparison-table"><thead><tr><th>Counted as</th><th>All templates</th><th>Requested utility</th><th>Sample</th></tr></thead><tbody>
    <tr><td><strong>Records</strong><p class="excerpt">What production receives, repeats included</p></td><td>${pct(rel.record_level.all)}</td><td>${pct(rel.record_level.requested_utility)}</td><td>${count(rel.record_level.samples)}</td></tr>
    <tr><td><strong>Families</strong><p class="excerpt">Each distinct wording counted once</p></td><td>${pct(rel.family_level.all)}</td><td>${pct(rel.family_level.requested_utility)}</td><td>${count(rel.family_level.samples)}</td></tr>
    </tbody></table></div>
    <div class="notice">${escapeHtml(rel.note)}</div>
    <dl class="facts"><div><dt>${cv.folds}-fold cross-validation (${escapeHtml(cv.scope)})</dt><dd>${pct(cv.mean)} &plusmn; ${(cv.sd * 100).toFixed(1)}pp</dd></div></dl>
    <p class="small">The fold spread is tight, so the plateau is a property of the data rather than of one unlucky split.</p>`;
}

async function loadOutputs() {
  const host = $('#conversion-outputs');
  if (!host) return;
  let rows;
  try {
    const response = await fetch('conversions.json');
    if (!response.ok) return;
    rows = await response.json();
  } catch (error) { return; }
  if (!rows?.length) return;
  host.innerHTML = `<h3 style="margin-top:32px">The converted templates</h3>
    <p class="small">Every rewrite the classifier ratified, strongest movement first. This is real message text.</p>
    ${rows.map((r, i) => `<details class="conversion"${i === 0 ? ' open' : ''}>
      <summary><strong>${escapeHtml(r.name || 'Untitled')}</strong> <span class="pill green">${pct(r.before)} &rarr; ${pct(r.after)}</span>${r.placeholders_lost?.length ? ` <span class="pill amber">lost ${escapeHtml(r.placeholders_lost.join(', '))}</span>` : ''}</summary>
      <div class="pair"><div><h4>Before &mdash; recorded MARKETING</h4><pre class="body-text">${escapeHtml(r.before_body)}</pre></div>
      <div><h4>After &mdash; utility candidate</h4><pre class="body-text">${escapeHtml(r.after_body)}</pre></div></div>
      ${r.split_off ? `<h4>Split off as a separate marketing template</h4><pre class="body-text split">${escapeHtml(r.split_off)}</pre>` : ''}
      <h4>Utility template JSON</h4><pre class="json">${escapeHtml(JSON.stringify(r.utility_json, null, 2))}</pre>
      ${r.split_off_json ? `<h4>Marketing template JSON</h4><pre class="json">${escapeHtml(JSON.stringify(r.split_off_json, null, 2))}</pre>` : ''}
      <p class="small">${escapeHtml(r.method || '')}</p>
    </details>`).join('')}`;
  paintIcons();
}

function conversionHtml(c) {
  const share = c.templates ? c.ratified_as_utility / c.templates : 0;
  const order = ['SPLIT_RECOMMENDED', 'ALREADY_UTILITY', 'NEEDS_CONTEXT', 'IRREDUCIBLY_MARKETING', 'ERROR'];
  const label = {
    SPLIT_RECOMMENDED: 'Converted and split',
    ALREADY_UTILITY: 'No promotional wording found',
    NEEDS_CONTEXT: 'Could not be separated safely',
    IRREDUCIBLY_MARKETING: 'Nothing transactional to keep',
    ERROR: 'Failed',
  };
  return `<div class="metrics">${metric('Templates attempted', count(c.templates), escapeHtml(c.scope || ''))}${metric('Rewrites produced', count(c.produced_a_rewrite), 'Passed every safety guard')}${metric('Ratified as utility', count(c.ratified_as_utility), pct(share) + ' of those attempted')}${metric('Average score gain', c.mean_score_gain == null ? 'n/a' : `+${(c.mean_score_gain * 100).toFixed(1)}pp`, 'On ratified conversions only')}</div>
    <div class="notice blue">A conversion counts only when the trained classifier reads the result as utility. Counting rewrites instead would reward a permissive filter: a checklist can only see wording it has vocabulary for.</div>
    <div class="grid-two"><div><h3>What happened to each template</h3><div class="table-wrap"><table class="comparison-table"><thead><tr><th>Outcome</th><th>Templates</th><th>Share</th></tr></thead><tbody>${order.filter(k => c.verdicts?.[k]).map(k => `<tr><td>${escapeHtml(label[k] || k)}</td><td>${count(c.verdicts[k])}</td><td>${pct(c.verdicts[k] / c.templates)}</td></tr>`).join('')}</tbody></table></div></div>
    <div><h3>Safety</h3><dl class="facts"><div><dt>Rewrites the classifier rejected</dt><dd>${count(c.produced_a_rewrite - c.ratified_as_utility)}</dd></div><div><dt>Conversions that lost a placeholder</dt><dd>${count(c.lost_a_placeholder)}</dd></div><div><dt>Failed requests</dt><dd>${count(c.failures)}</dd></div></dl>
    <p class="small">${escapeHtml(c.limitation || '')}</p></div></div>`;
}

function slicesHtml(slices) {
  const rows = [['requested_utility', 'Requested utility', 'The live decision: will Meta downgrade this draft?'],
                ['all', 'All holdout families', 'Includes the trivially-correct cases below'],
                ['requested_marketing', 'Requested marketing', 'Meta never upgraded one of these, so the model cannot be wrong in a costly direction']];
  return `<h3 style="margin-top:28px">Performance by what was requested</h3>
    <div class="table-wrap"><table class="comparison-table"><thead><tr><th>Slice</th><th>Families</th><th>Accuracy</th><th>Majority</th><th>Utility precision</th></tr></thead><tbody>${rows.map(([key, label, note]) => {
      const s = slices[key];
      if (!s) return '';
      const strong = key === 'requested_utility';
      return `<tr><td><strong>${escapeHtml(label)}</strong><p class="excerpt">${escapeHtml(note)}</p></td><td>${count(s.samples)}</td><td>${strong ? `<strong>${pct(s.accuracy)}</strong>` : pct(s.accuracy)}</td><td>${pct(s.majority_accuracy)}</td><td>${s.utility_precision === null ? 'n/a' : (strong ? `<strong>${pct(s.utility_precision)}</strong>` : pct(s.utility_precision))}</td></tr>`;
    }).join('')}</tbody></table></div>`;
}

function bandHtml(bands, thresholds) {
  return `<h3 style="margin-top:28px">Referring the uncertain cases</h3>
    <p class="small">Below ${thresholds ? pct(thresholds.marketing) : ''} the model answers marketing, above ${thresholds ? pct(thresholds.utility) : ''} utility, and in between it refers the template for human review instead of guessing. Thresholds come from out-of-fold training predictions, never the holdout.</p>
    <div class="metrics">${metric('Referred for review', count(bands.referred), `${pct(1 - bands.coverage)} of the holdout`)}${metric('Answered', count(bands.decided), pct(bands.coverage) + ' coverage')}${metric('Accuracy when answered', bands.accuracy_when_decided === null ? 'n/a' : pct(bands.accuracy_when_decided), 'Excludes referred templates')}${metric('Utility precision when answered', bands.utility_precision_when_decided === null ? 'n/a' : pct(bands.utility_precision_when_decided), 'Target was 85%')}</div>`;
}

function experimentsHtml(e) {
  const r = e.report;
  return `${e.stale ? '<div class="notice">This experiment belongs to an older dataset revision. Rerun it in the main workspace.</div>' : ''}
    <div class="notice blue">These are different test examples from the random holdout above. Compare models within this experiment, not as a before-and-after improvement over the original baseline.</div>
    <dl class="facts"><div><dt>Chronological cutoff</dt><dd>${escapeHtml(r.split.cutoff)}</dd></div><div><dt>Training / test families</dt><dd>${count(r.split.train_families)} / ${count(r.split.test_families)}</dd></div><div><dt>Boundary families purged / invalid-date exclusions</dt><dd>${count(r.split.purged_boundary_families)} / ${count(r.split.excluded_invalid_date_families)}</dd></div><div><dt>High lexical overlap / normalized family overlap</dt><dd>${count(r.near_duplicates.flagged_test_families)} / ${count(r.split.group_overlap)}</dd></div></dl>
    <div class="comparison-controls"><label for="comparison-subset">Evaluation subset</label><select id="comparison-subset"><option value="all">Full chronological holdout</option><option value="lower_overlap">Lower lexical overlap only</option></select></div><div id="comparison-data"></div>
    <details><summary>Experiment methodology and limitations</summary><p class="small">Generated: ${escapeHtml(r.generated_at)}. Training-majority accuracy on the full holdout: ${pct(r.majority_accuracy)}.</p><ul>${r.limitations.map(text => `<li>${escapeHtml(text)}</li>`).join('')}</ul></details>`;
}

function renderComparison() {
  const rows = report.experiments.report.results;
  $('#comparison-data').innerHTML = `<div class="table-wrap"><table class="comparison-table"><thead><tr><th>Method</th><th>Test examples</th><th>Utility precision</th><th>Utility recall</th><th>False utility</th><th>Accuracy</th></tr></thead><tbody>${rows.map(row => {
    const m = row[subset];
    return `<tr><td><strong>${escapeHtml(row.method)}</strong></td>${m ? `<td>${count(m.samples)}</td><td>${pct(m.utility_precision)}</td><td>${pct(m.utility_recall)}</td><td>${count(m.confusion_matrix[0][1])}</td><td>${pct(m.accuracy)}</td>` : '<td colspan="5">No examples in this subset</td>'}</tr>`;
  }).join('')}</tbody></table></div>`;
}

function renderErrors() {
  const rows = report.errors.items.filter(row => (direction === 'all' || row.direction === direction)
    && (reviewStatus === 'all' || row.annotation.status === reviewStatus)
    && [row.name,row.header,row.body,row.footer,row.buttons].join(' ').toLowerCase().includes(query.toLowerCase()));
  const pages = Math.max(1, Math.ceil(rows.length / 10));
  errorPage = Math.min(errorPage, pages);
  const shown = rows.slice((errorPage - 1) * 10, errorPage * 10);
  $('#error-table').innerHTML = `${shown.length ? `<div class="table-wrap"><table class="error-table"><thead><tr><th>Template</th><th>Meta category</th><th>Prediction</th><th>Review status</th></tr></thead><tbody>${shown.map(row => `<tr><td><button class="error-name" data-id="${escapeHtml(row.id)}" title="${escapeHtml(row.name)}">${escapeHtml(row.name || 'Untitled template')}</button><p class="excerpt">${escapeHtml(row.body)}</p></td><td>${badge(row.recorded_category)}</td><td>${badge(row.prediction)}</td><td>${escapeHtml(row.annotation.status.replaceAll('_',' '))}</td></tr>`).join('')}</tbody></table></div>` : '<div class="empty">No error examples match these filters.</div>'}
    <div class="pagination"><span>${count(rows.length)} examples / Page ${errorPage} of ${pages}</span><div class="actions"><button id="previous-errors" title="Previous errors" aria-label="Previous errors" ${errorPage === 1 ? 'disabled' : ''}>${icon('chevron-left')}</button><button id="next-errors" title="Next errors" aria-label="Next errors" ${errorPage === pages ? 'disabled' : ''}>${icon('chevron-right')}</button></div></div>`;
  $('#previous-errors').onclick = () => { errorPage--; renderErrors(); };
  $('#next-errors').onclick = () => { errorPage++; renderErrors(); };
  document.querySelectorAll('[data-id]').forEach(button => { button.onclick = () => openTemplate(button.dataset.id); });
  paintIcons();
}

function openTemplate(id) {
  const row = report.errors.items.find(item => item.id === id);
  if (!row) return;
  $('#dialog-title').textContent = row.name || 'Untitled template';
  const a = row.annotation;
  $('#dialog-content').innerHTML = `<p style="margin-top:16px">Meta ${badge(row.recorded_category)} / Prediction ${badge(row.prediction)}</p>${['header','body','footer','buttons'].map(key => `<div class="component"><h3>${key[0].toUpperCase() + key.slice(1)}</h3><p>${escapeHtml(row[key] || '(empty)')}</p></div>`).join('')}
    <div class="component"><h3>Human review</h3><p>Status: ${escapeHtml(a.status.replaceAll('_',' '))}</p><p>Reason: ${escapeHtml(a.reason.replaceAll('_',' '))}</p><p>${escapeHtml(a.notes || 'No review notes yet.')}</p></div><div class="component"><h3>Business context</h3><p>${escapeHtml(a.context || 'Not supplied.')}</p></div>
    ${a.draft ? `<div class="component"><h3>Saved candidate draft</h3>${['header','body','footer','buttons'].map(key => `<p><strong>${key}:</strong> ${escapeHtml(a.draft[key] || '(empty)')}</p>`).join('')}</div>` : ''}
    <div class="component"><h3>Manually recorded Meta outcome</h3><p>${escapeHtml(a.meta_outcome.replaceAll('_',' '))}</p><p>${escapeHtml(a.outcome_reference || '')}</p></div>`;
  $('#template-dialog').showModal();
}

$('#refresh').onclick = loadResults;
$('#close-dialog').onclick = () => $('#template-dialog').close();
$('#download').onclick = () => {
  if (!report) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type:'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'template-lab-results.json';
  document.body.appendChild(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
paintIcons();
loadResults();
