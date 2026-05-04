#!/usr/bin/env python3
"""Single Haiku 4.5 call for Stage C paraphrase rounds."""
import json, sys, urllib.request, urllib.error

CRED = json.load(open('/root/.claude/.credentials.json'))
TOKEN = CRED['claudeAiOauth']['accessToken']

API = 'https://api.anthropic.com/v1/messages'
SYS = "You are Claude Code, Anthropic's official CLI for Claude."

def call(model, prompt, max_tokens=2048):
    body = json.dumps({
        'model': model,
        'max_tokens': max_tokens,
        'system': SYS,
        'messages': [{'role': 'user', 'content': prompt}],
    }).encode()
    req = urllib.request.Request(API, data=body, headers={
        'Authorization': f'Bearer {TOKEN}',
        'anthropic-version': '2023-06-01',
        'anthropic-beta': 'oauth-2025-04-20',
        'content-type': 'application/json',
    })
    try:
        resp = urllib.request.urlopen(req, timeout=300)
        data = json.loads(resp.read())
        return data['content'][0]['text']
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors='replace')
        if e.code == 429 or 'rate_limit' in body.lower() or 'rate-limit' in body.lower():
            print(f'RATE_LIMIT: {e.code} {body[:300]}', file=sys.stderr)
            sys.exit(42)
        print(f'HTTP_ERROR {e.code}: {body[:500]}', file=sys.stderr)
        raise

if __name__ == '__main__':
    model = sys.argv[1]  # claude-haiku-4-5
    prompt_file = sys.argv[2]
    out_file = sys.argv[3]
    prompt = open(prompt_file).read()
    out = call(model, prompt)
    open(out_file, 'w').write(out)
    print(f'wrote {out_file} ({len(out)} chars)')
