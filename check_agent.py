import ast, sys

path = r'C:\Users\Lenovo\Desktop\Everything\oop cp\Heart-Disease-Prediction-System\health\ai_calling_agent.py'
with open(path, encoding='utf-8') as f:
    src = f.read()

if 'google.generativeai' in src or 'genai.configure' in src or 'genai.GenerativeModel' in src:
    print('FAIL: Gemini references still present')
    sys.exit(1)

if 'api.quatarly.cloud' not in src or 'CLAUDE_MODEL' not in src:
    print('FAIL: Claude API references missing')
    sys.exit(1)

ast.parse(src)
print('OK: No Gemini references, Claude API wired in, syntax valid')
