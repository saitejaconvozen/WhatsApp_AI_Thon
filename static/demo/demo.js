const form = document.querySelector('#template-form');
const result = document.querySelector('#result');
const button = document.querySelector('#classify');
window.lucide?.createIcons();
form.onreset = () => { result.replaceChildren(Object.assign(document.createElement('p'), { className: 'muted', textContent: 'No prediction yet.' })); };
form.onsubmit = async event => {
  event.preventDefault();
  button.disabled = true;
  button.querySelector('span').textContent = 'Classifying...';
  result.textContent = 'Calculating prediction...';
  try {
    const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(Object.fromEntries(new FormData(form))) });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Check the template fields and try again.');
    const category = document.createElement('p');
    category.className = `category ${data.category === 'UTILITY' ? 'UTILITY' : 'MARKETING'}`;
    category.textContent = data.category;
    const details = document.createElement('dl');
    const utilityScore = data.utility_probability == null ? 'Not applicable' : `${(data.utility_probability * 100).toFixed(1)}%`;
    for (const [label, value] of [[data.score_label || 'Utility score', utilityScore], ['Review band', data.band || 'Not available'], ['Model', data.model]]) {
      const row = document.createElement('div');
      row.append(Object.assign(document.createElement('dt'), { textContent: label }), Object.assign(document.createElement('dd'), { textContent: value }));
      details.append(row);
    }
    result.replaceChildren(category, details, Object.assign(document.createElement('p'), { className: 'notice', textContent: data.notice }));
  } catch (error) {
    result.replaceChildren(Object.assign(document.createElement('p'), { className: 'error', textContent: error.message }));
  } finally {
    button.disabled = false;
    button.querySelector('span').textContent = 'Classify template';
  }
};
