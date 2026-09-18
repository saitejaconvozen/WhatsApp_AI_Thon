/* Workbench for the five endpoints. Each renders its own shape; nothing is chained. */
const $ = (id) => document.getElementById(id);
const templateForm = $("template-form"), taskForm = $("task-form");
const result = $("result"), draft = $("draft");

const MODES = {
  classify: {path: "/api/predict",  label: "Classify",              icon: "scan-text",
             heading: "Predicted Meta category",
             hint: "Runs locally. No text leaves this machine."},
  explain:  {path: "/api/explain",  label: "Explain",               icon: "message-square-quote",
             heading: "Why this category",
             hint: "Asks the model to cite the policy clauses it applied. A few seconds."},
  convert:  {path: "/api/convert",  label: "Convert to utility",    icon: "repeat",
             heading: "Conversion",
             hint: "Rewrites and re-classifies until the classifier accepts it, or the rounds run out."},
  split:    {path: "/api/split",    label: "Split",                 icon: "split",
             heading: "Split result",
             hint: "For a template carrying both a service update and an offer: the offer moves to its own marketing template rather than being deleted."},
};
let mode = "classify";

const esc = (s) => String(s ?? "").replace(/[&<>]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const marked = (s) => esc(s).replace(/\{\{[^}]{0,60}\}\}/g, (m) => `<span class="ph">${m}</span>`);
const pct = (v) => (typeof v === "number" ? v.toFixed(3) : "—");
const icons = () => window.lucide && window.lucide.createIcons();

function rows(pairs) {
  const shown = pairs.filter(([, v]) => v !== null && v !== undefined && v !== "");
  if (!shown.length) return "";
  return `<dl>${shown.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl>`;
}

function templateCard(title, body, cls = "") {
  if (!body) return "";
  return `<div class="card ${cls}"><h3>${esc(title)}</h3><p class="body-text">${marked(body)}</p></div>`;
}

function jsonBlock(label, value) {
  if (!value) return "";
  return `<h3 class="sec">${esc(label)}</h3><pre class="json">${esc(JSON.stringify(value, null, 2))}</pre>`;
}

/* ---------- renderers, one per endpoint ---------- */

function renderClassify(d) {
  return `<p class="verdict ${esc(d.category)}">${esc(d.category)}</p>
    <p class="sub">${esc(d.score_label || "Utility probability")}: <b>${pct(d.utility_probability)}</b></p>
    ${rows([["Band", d.band], ["Model", d.model]])}
    <p class="meta-line">${esc(d.notice || "")}</p>`;
}

function renderExplain(d) {
  if (!d.available) {
    return `<p class="error">${esc(d.reason || "The reviewer is not available.")}</p>`;
  }
  const clauses = (d.clauses || [])
    .map((c) => `<li><strong>${esc(c.id)}</strong>${esc(c.text)}</li>`).join("");
  return `<p class="verdict ${esc(d.category)}">${esc(d.category)}</p>
    ${d.agrees === false
      ? `<p class="flag">The local classifier and this reading disagree. On measured disagreements
         each was right about half the time, so neither is authoritative here &mdash; a human should look.</p>`
      : ""}
    <p class="rationale">${esc(d.rationale || "")}</p>
    ${clauses ? `<h3 class="sec">Clauses applied</h3><ul class="clauses">${clauses}</ul>` : ""}
    <p class="meta-line">${esc(d.limitation || d.notice || "")}</p>`;
}

function renderConvert(d) {
  if (d.already_utility) {
    return `<p class="verdict UTILITY">Already utility</p>
      <p class="sub">No rewrite attempted &mdash; the classifier reads the template as utility at
        <b>${pct(d.confidence)}</b>.</p>
      ${templateCard("Template, unchanged", (d.template || {}).body, "good")}`;
  }
  const trail = (d.rounds || []).map((r) => {
    if (r.outcome === "refused") {
      return `<div class="round"><span class="n">round ${r.round}</span>
        <span class="blocked">model refused</span>
        <span class="issue">${esc(r.reason || "")}</span></div>`;
    }
    const blocked = r.gates_passed === false;
    return `<div class="round"><span class="n">round ${r.round}</span>
      <span class="cat ${esc(r.category)}">${esc(r.category)}</span>
      <span class="n">${pct(r.confidence)}</span>
      ${blocked ? `<span class="blocked">gates blocked</span>` : ""}
      ${(r.issues || []).slice(0, 2).map((i) => `<span class="issue">${esc(i)}</span>`).join("")}</div>`;
  }).join("");

  if (!d.converted) {
    return `<p class="verdict MARKETING">Not converted</p>
      <p class="sub">${esc(d.reason || "")}</p>
      ${trail ? `<h3 class="sec">Rounds tried</h3><div class="trail">${trail}</div>` : ""}`;
  }
  return `<p class="verdict UTILITY">Converted</p>
    <p class="sub">Utility score <b>${pct(d.confidence_before)}</b> &rarr; <b>${pct(d.confidence)}</b>
      over ${(d.rounds || []).length} round${(d.rounds || []).length === 1 ? "" : "s"}.</p>
    ${templateCard("Rewritten template", (d.template || {}).body, "good")}
    ${(d.reasoning || []).map((r) => `<p class="meta-line">${esc(r)}</p>`).join("")}
    ${trail ? `<h3 class="sec">Rounds</h3><div class="trail">${trail}</div>` : ""}
    ${jsonBlock("Template JSON", d.template_json)}
    <p class="meta-line">${esc(d.notice || "")}</p>`;
}

function renderSplit(d) {
  const findings = (d.findings || [])
    .map((f) => `<li><strong>${esc(f.code)}</strong>${esc(f.message)}</li>`).join("");
  const missing = (d.missing_context || [])
    .map((m) => `<li>${esc(m)}</li>`).join("");
  return `<p class="verdict ${d.utility ? "UTILITY" : "NEUTRAL"}">${esc((d.verdict || "").replace(/_/g, " "))}</p>
    <p class="sub">${esc(d.reason || "")}</p>
    ${templateCard("Utility part", (d.utility || {}).body, "good")}
    ${templateCard("Moved to a marketing template", (d.split_off || {}).body, "promo")}
    ${findings ? `<h3 class="sec">Promotional findings</h3><ul class="clauses">${findings}</ul>` : ""}
    ${missing ? `<h3 class="sec">Context still needed</h3><ul class="clauses">${missing}</ul>` : ""}
    ${jsonBlock("Utility template JSON", d.utility_json)}
    ${jsonBlock("Marketing template JSON", d.split_off_json)}
    <p class="meta-line">${esc(d.notice || "")}</p>`;
}

function renderDraft(d) {
  if (!d.possible) {
    return `<p class="verdict MARKETING">Declined</p>
      <p class="rationale">${esc(d.reason || "")}</p>
      <p class="meta-line">A promotional task is refused rather than dressed up as a service
        message. Describe an event that has already happened to the recipient.</p>`;
  }
  const t = d.template || {};
  // The form's approval rate is shown only when the draft actually belongs to that
  // form; below the match threshold the API returns null and there is no rate to quote.
  const evidence = d.form_matched && typeof d.form_approval_rate === "number"
    ? `${(d.form_approval_rate * 100).toFixed(0)}% of templates in this form were approved`
    : "No closely matching approved form";
  return `<p class="verdict UTILITY">${esc(t.name || "Drafted")}</p>
    ${t.buttons ? `<p class="sub">Buttons: ${esc(t.buttons)}</p>` : ""}
    ${templateCard("Draft", t.body, "good")}
    ${rows([["Utility score", pct(d.score)], ["Precedent", evidence], ["Method", d.method]])}
    ${d.evidence ? `<p class="meta-line">${esc(d.evidence)}</p>` : ""}
    ${d.reason ? `<p class="meta-line">${esc(d.reason)}</p>` : ""}
    ${jsonBlock("Template JSON", d.template_json)}
    <p class="meta-line">${esc(d.notice || "")}</p>`;
}

const RENDER = {classify: renderClassify, explain: renderExplain,
                convert: renderConvert, split: renderSplit};

/* ---------- mode switching ---------- */

function setMode(next) {
  mode = next;
  const generating = next === "generate";
  $("template-workspace").hidden = generating;
  $("task-workspace").hidden = !generating;
  document.querySelectorAll(".modes button").forEach((b) =>
    b.setAttribute("aria-selected", String(b.dataset.mode === next)));
  if (generating) return;

  const spec = MODES[next];
  $("output-heading").textContent = spec.heading;
  $("mode-hint").textContent = spec.hint;
  const button = $("submit-template");
  button.querySelector("span").textContent = spec.label;
  button.querySelector("i,svg")?.setAttribute("data-lucide", spec.icon);
  // Only conversion loops, and only splitting asks about the recipient relationship.
  $("rounds-field").hidden = next !== "convert";
  $("relationship-field").hidden = next !== "split";
  result.innerHTML = `<p class="muted">Nothing yet.</p>`;
  icons();
}

document.querySelectorAll(".modes button").forEach((b) =>
  b.addEventListener("click", () => setMode(b.dataset.mode)));

$("json-mode").addEventListener("change", (e) => {
  $("json-field").hidden = !e.target.checked;
  $("plain-fields").hidden = e.target.checked;
  templateForm.body.required = !e.target.checked;
});

/* ---------- submission ---------- */

async function send(path, payload, target, render, button, busyLabel) {
  const span = button.querySelector("span"), original = span.textContent;
  button.disabled = true; span.textContent = busyLabel;
  target.innerHTML = `<p class="muted">Working…</p>`;
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      // FastAPI returns `detail` as a string for our own errors and an array of
      // field objects for schema validation, so both shapes have to render.
      const detail = data && data.detail;
      const message = Array.isArray(detail)
        ? detail.map((d) => `${(d.loc || []).slice(-1)}: ${d.msg}`).join("; ")
        : (detail || `Request failed (${response.status}).`);
      target.innerHTML = `<p class="error">${esc(message)}</p>`;
      return;
    }
    target.innerHTML = render(data);
  } catch (error) {
    target.innerHTML = `<p class="error">${esc(error.message || "Could not reach the server.")}</p>`;
  } finally {
    button.disabled = false; span.textContent = original; icons();
  }
}

templateForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const form = new FormData(templateForm);
  const payload = {};
  for (const [k, v] of form.entries()) {
    if (k === "max_rounds") payload[k] = Number(v);
    else if (k === "relationship_confirmed") payload[k] = true;
    else if (v) payload[k] = v;
  }
  if ($("json-mode").checked) {
    ["header", "body", "footer", "buttons", "requested_category"].forEach((k) => delete payload[k]);
  } else {
    delete payload.template_json;
  }
  if (mode !== "convert") delete payload.max_rounds;
  send(MODES[mode].path, payload, result, RENDER[mode], $("submit-template"),
       mode === "classify" ? "Classifying…" : "Working…");
});

taskForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const form = new FormData(taskForm);
  const payload = {task: form.get("task")};
  if (form.get("context")) payload.context = form.get("context");
  send("/api/generate", payload, draft, renderDraft, $("submit-task"), "Drafting…");
});

templateForm.addEventListener("reset", () => {
  result.innerHTML = `<p class="muted">Nothing yet.</p>`;
});
taskForm.addEventListener("reset", () => {
  draft.innerHTML = `<p class="muted">Nothing yet.</p>`;
});

setMode("classify");
