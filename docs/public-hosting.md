# Public Model Demo

Live link created on 2026-09-18:

https://impressive-guam-graphs-architecture.trycloudflare.com

The root URL now opens the interactive classifier, verified with a real public
prediction on desktop and mobile. The aggregate dashboard is at `/results/`.
The second hostname `servers-centered-approve-installations.trycloudflare.com`
failed DNS checks; do not use it.

This is a temporary Cloudflare Quick Tunnel, not a permanent cloud deployment.
It forwards HTTPS traffic to an inference-only FastAPI server on port 8769. The local
computer, static server and tunnel must remain running. Restarting a Quick
Tunnel normally creates a different URL. No uptime guarantee is assumed.

## Public Boundary

`templatelab/demo.py` exposes only `/api/predict`, the classifier UI and the
aggregate `site/` directory under `/results/`. Prediction output is allowlisted:
category, score, review band and fixed model/limitation text. Stored neighbors,
customer messages, identifiers, annotations, credentials, training routes and
the private editor are not exposed. Inputs are not persisted or sent to a hosted
AI provider. The endpoint has field-size limits and a global 60-request/minute
limit. This is not a production-authenticated service.

Browser checks verified desktop/mobile predictions and 404s for `/api/records`,
`/api/export`, `/.env` and `/.data/templates.sqlite3` at the public URL.

The public site is a snapshot. Refreshing it reloads the last generated report;
it does not read the private database directly. Update it locally with:

```bash
.venv/bin/python -m templatelab.publish
```

Current server commands:

```bash
.venv/bin/python -m uvicorn templatelab.demo:app --host 127.0.0.1 --port 8769
/tmp/templatelab-cloudflared tunnel --url http://127.0.0.1:8769 --no-autoupdate --protocol http2
```

The official Cloudflare client was downloaded from the Cloudflare GitHub
release. Its reported version is 2026.9.1. The demo serves the existing baseline;
experimental fine-tuned models are not promoted automatically.

Permanent hosting requires connecting a Cloudflare account/project, or a
GitHub repository for the existing Pages workflow. Do not tunnel the private
FastAPI app: it has no production authentication or multi-user authorization.

Reference: [Cloudflare Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/).
