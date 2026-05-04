#!/usr/bin/env python3
"""Single Pollinations call for Stage C R13_C paraphrase."""
import json, sys
from urllib import request as ureq

prompt_file = sys.argv[1]
out_file = sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "openai-fast"

prompt = open(prompt_file).read()

payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 4000,
    "temperature": 0.85,
}
req = ureq.Request(
    "https://text.pollinations.ai/openai",
    data=json.dumps(payload).encode(),
    headers={
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    },
)
with ureq.urlopen(req, timeout=240) as r:
    body = json.loads(r.read())
msg = body["choices"][0]["message"]
out = msg.get("content") or msg.get("reasoning_content") or ""
if not out.strip():
    print("EMPTY content. full message: " + json.dumps(msg)[:1500], file=sys.stderr)
    sys.exit(2)
open(out_file, "w").write(out)
print(f"wrote {out_file} ({len(out)} chars)")
