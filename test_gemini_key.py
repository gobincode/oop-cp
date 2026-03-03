import urllib.request, json

API_KEY = 'your-api-key-1'
MODEL = 'claude-haiku-4-5-20251001'
BASE_URL = 'https://api.quatarly.cloud'

# Anthropic-style endpoint: POST /v1/messages
payload = json.dumps({
    'model': MODEL,
    'max_tokens': 20,
    'messages': [{'role': 'user', 'content': 'Reply with just: OK'}]
}).encode()

headers = {
    'Content-Type': 'application/json',
    'x-api-key': API_KEY,
    'anthropic-version': '2023-06-01'
}

url = f'{BASE_URL}/v1/messages'
print(f'POST {url}')
req = urllib.request.Request(url, data=payload, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read())
        print(f'Status: 200 OK')
        print(f'Response: {json.dumps(data, indent=2)}')
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f'Status: {e.code}')
    print(f'Response: {body}')
except Exception as e:
    print(f'ERROR: {e}')
