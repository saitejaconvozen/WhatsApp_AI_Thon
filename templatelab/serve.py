"""Launch the local app, optionally prompting privately for a hosted LLM key."""

import argparse
import getpass
import os
from urllib.parse import urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--model", default="deepseek-v4-flash-fireworks-api")
    parser.add_argument("--base-url", default="https://litellm-proxy-1099428089593.asia-south1.run.app")
    parser.add_argument("--allow-egress", action="store_true")
    args = parser.parse_args()
    if args.llm:
        url = urlsplit(args.base_url)
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
            parser.error("Use an HTTPS provider URL without embedded credentials, queries or fragments.")
        if not args.allow_egress:
            parser.error("--allow-egress is required: templates, context and examples may be sent to the provider.")
        key = getpass.getpass("Provider API key (not saved): ").strip()
        if not key:
            parser.error("An API key is required.")
        os.environ.update({"TEMPLATELAB_LLM_BACKEND": "openai_compatible",
                           "TEMPLATELAB_LLM_ALLOW_EGRESS": "1",
                           "TEMPLATELAB_LLM_MODEL": args.model,
                           "TEMPLATELAB_LLM_BASE_URL": args.base_url,
                           "TEMPLATELAB_LLM_API_KEY": key,
                           "TEMPLATELAB_LLM_MAX_TOKENS": "8192"})
        print("Hosted AI enabled for explicit review, conversion and generation requests. Provider charges may apply.")
    else:
        os.environ["TEMPLATELAB_LLM_BACKEND"] = "disabled"
    import uvicorn
    uvicorn.run("templatelab.app:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
