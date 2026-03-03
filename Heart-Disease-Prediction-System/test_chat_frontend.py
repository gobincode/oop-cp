"""
End-to-end frontend simulation for the AI Medical Chat.
Uses Django test client to POST exactly as the browser would.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'health_desease.settings')
import django
django.setup()

from django.test import Client
from django.contrib.auth.models import User
from health.models import Patient, Search_Data, ECG_Prediction
import datetime

# ── helpers ────────────────────────────────────────────────────────────────

def get_or_create_test_patient():
    """Return (user, patient) - use first real patient if exists, else create one."""
    patient = Patient.objects.first()
    if patient:
        return patient.user, patient

    user = User.objects.create_user(
        username='test_chat_user', password='testpass123',
        first_name='Test', last_name='Patient'
    )
    patient = Patient.objects.create(
        user=user,
        contact='9999999999',
        address='123 Test St',
        dob=datetime.date(1985, 6, 15)
    )
    # Fake prediction record so context is rich
    Search_Data.objects.create(
        patient=patient,
        result='AT RISK',
        prediction_accuracy='83.12',
        values_list='[57, 1, 0, 150, 276, 0, 0, 112, 1, 3.5, 0, 2, 3]'
    )
    return user, patient


def send_message(client, message, session_id=None):
    resp = client.post(
        '/medical_chat_send',
        data=json.dumps({'message': message, 'session_id': session_id}),
        content_type='application/json'
    )
    ct = resp.get('Content-Type', '')
    if 'json' in ct:
        return resp.status_code, resp.json()
    return resp.status_code, {'error': f'non-json response (status {resp.status_code})'}


def check(label, status, data):
    ok = status == 200 and 'reply' in data and len(data['reply']) > 20
    icon = 'PASS' if ok else 'FAIL'
    reply_preview = data.get('reply', data.get('error', ''))[:120].replace('\n', ' ')
    reply_preview = reply_preview.encode('ascii', errors='replace').decode('ascii')
    print(f"[{icon}] {label}")
    print(f"       Status: {status}  |  Reply: {reply_preview}...")
    print()
    return ok


# ── run tests ──────────────────────────────────────────────────────────────

print("=" * 70)
print("AI MEDICAL CHAT — FRONTEND ENDPOINT TEST")
print("=" * 70 + "\n")

user, patient = get_or_create_test_patient()
print(f"Using patient: {user.username}  (id={patient.id})")
print(f"Has prediction: {Search_Data.objects.filter(patient=patient).exists()}")
print()

client = Client()
client.force_login(user)

results = []
session_id = None

# 1. Ask about the prediction result (core use case)
status, data = send_message(client, "What does my AT RISK result mean?")
results.append(check("Q1: Explain AT RISK result", status, data))
if 'session_id' in data:
    session_id = data['session_id']

# 2. Follow-up in same session (tests conversation history)
status, data = send_message(client, "What lifestyle changes should I make?", session_id)
results.append(check("Q2: Lifestyle advice (same session)", status, data))

# 3. Diet question
status, data = send_message(client, "What foods should I avoid?", session_id)
results.append(check("Q3: Diet advice", status, data))

# 4. Exercise question
status, data = send_message(client, "How much exercise is recommended for someone like me?", session_id)
results.append(check("Q4: Exercise guidance", status, data))

# 5. Doctor referral question
status, data = send_message(client, "Should I see a cardiologist?", session_id)
results.append(check("Q5: Doctor referral question", status, data))

# 6. New session (tests session isolation)
status, data = send_message(client, "What is cholesterol and why does it matter?")
results.append(check("Q6: General heart health question (new session)", status, data))
new_session_id = data.get('session_id')

# 7. Stress question
status, data = send_message(client, "How does stress affect heart disease risk?", new_session_id)
results.append(check("Q7: Stress and heart health", status, data))

# 8. Test unauthenticated access is blocked
anon_client = Client()
status, data = send_message(anon_client, "What does my result mean?")
blocked = status in (302, 403)
print(f"[{'PASS' if blocked else 'FAIL'}] Q8: Unauthenticated access blocked (status={status})")
print()
results.append(blocked)

# 9. Empty message rejected
status, data = send_message(client, "   ")
rejected = status == 400
print(f"[{'PASS' if rejected else 'FAIL'}] Q9: Empty message rejected (status={status})")
print()
results.append(rejected)

# ── summary ────────────────────────────────────────────────────────────────
passed = sum(results)
total = len(results)
print("=" * 70)
print(f"RESULT: {passed}/{total} tests passed")
if passed == total:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED — check output above")
print("=" * 70)
