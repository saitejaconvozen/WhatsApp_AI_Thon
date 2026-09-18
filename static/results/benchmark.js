function pilotMarkup(data) {
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const percent = value => value == null ? 'Not available' : `${(value * 100).toFixed(1)}%`;
  if (!data?.available) return '<p class="muted">No DeepSeek pilot report is available yet.</p>';
  const r = data.report, m = r.metrics;
  const card = (label, value) => `<div><small>${label}</small><strong>${value}</strong></div>`;
  return `<p class="source-note">${escape(r.model)} | ${escape(r.completed)} / ${escape(r.max_calls)} templates completed | ${r.complete ? 'Complete' : 'Partial run; no final result'} | ${escape(r.generated_at)}</p>
    ${r.stale ? '<div class="notice warning">This report belongs to an older dataset revision.</div>' : ''}
    ${m ? `<div class="pilot-metrics">${card('End-to-end accuracy', percent(m.end_to_end?.accuracy))}${card('Baseline, same sample', percent(m.baseline_same_sample?.accuracy))}${card('Utility precision', percent(m.end_to_end?.utility_precision))}${card('Failures / abstentions', `${m.failures} / ${m.abstentions}`)}</div>
    <p>95% accuracy interval: ${m.accuracy_95pct_interval ? m.accuracy_95pct_interval.map(percent).join(' to ') : 'Not available'}. Every failed response and abstention counts as incorrect.</p>
    <div class="notice">${!r.complete ? 'The pilot is incomplete. Do not interpret partial metrics as the final result.' : m.target_observed ? 'The observed pilot accuracy reached 90%. This small historical sample does not establish 90% future accuracy.' : 'The observed pilot accuracy did not reach 90%. No serving model has been replaced.'}</div>` : '<p>Waiting for the first provider response.</p>'}
    <p class="muted">${escape(r.selection)}. ${escape(r.retrieval)}. Maximum ${r.max_calls} calls; retries: ${r.retries}.</p>
    <details><summary>Evaluation limitations</summary><ul>${r.limitations.map(item => `<li>${escape(item)}</li>`).join('')}</ul></details>`;
}

async function renderBenchmark() {
  document.querySelector('#app').innerHTML = heading('DeepSeek evaluation', 'Meta-label prediction / requested-utility pilot', '<button class="icon-button" id="refresh-pilot" title="Refresh pilot" aria-label="Refresh pilot"><i data-lucide="refresh-cw"></i></button>') + '<div id="pilot-report">Loading...</div>';
  icons();
  document.querySelector('#refresh-pilot').onclick = renderBenchmark;
  try {
    const data = await api('/api/benchmark');
    if (state.route === 'benchmark') document.querySelector('#pilot-report').innerHTML = pilotMarkup(data);
  } catch (error) { if (state.route === 'benchmark') document.querySelector('#pilot-report').textContent = error.message; }
}

function improvementMarkup(data) {
  if (!data?.available) return '<p>No model-improvement report is available.</p>';
  const pct = value => value == null ? 'Pending' : `${(value * 100).toFixed(1)}%`;
  return '<p>Validation results select models; they are not final test accuracy. The 90% target remains unverified.</p>' +
    data.stages.map(stage => `<div class="pilot-metrics"><div><small>${stage.name}</small><strong>${stage.complete ? 'Complete' : 'Incomplete'}</strong></div><div><small>Requested-utility validation</small><strong>${pct(stage.selected_validation?.requested_utility?.accuracy)}</strong></div><div><small>All validation</small><strong>${pct(stage.selected_validation?.all?.accuracy)}</strong></div><div><small>Configurations / checkpoints</small><strong>${Number(stage.candidates)}</strong></div></div>`).join('');
}
