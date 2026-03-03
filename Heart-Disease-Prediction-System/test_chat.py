"""Quick test: Claude responds correctly via the chat helper."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'health_desease.settings')
import django; django.setup()

from health.views import _call_claude_chat, CHAT_SYSTEM_PROMPT

patient_ctx = "PATIENT: TestUser\nAGE: 45\nLATEST HEART PREDICTION: AT RISK (accuracy 83.12%)\nINPUT VALUES: [45, 1, 0, 150, 276, 0, 0, 112, 1, 3.5, 0, 2, 3]"
system = CHAT_SYSTEM_PROMPT.format(patient_context=patient_ctx)
messages = [{"role": "user", "content": "What does my AT RISK result mean?"}]

reply = _call_claude_chat(system, messages, os.getenv('CLAUDE_API_KEY', 'your-api-key-1'), max_tokens=500)
print("AI Reply:", reply[:300])
print("\nOK - Chat endpoint works")
