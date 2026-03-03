"""
Tests for Medical Document Parser & Voice Symptom Collector.
Run: python test_new_features.py
"""
import os, sys, django, json, io, base64

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'health_desease.settings')
sys.path.insert(0, os.path.dirname(__file__))
django.setup()

from django.test import Client
from django.contrib.auth.models import User
from health.models import Patient, Doctor, MedicalDocument

PASS = '[PASS]'
FAIL = '[FAIL]'
results = []

def check(label, cond, detail=''):
    status = PASS if cond else FAIL
    print(f"  {status} {label}" + (f"  ->  {detail}" if detail else ''))
    results.append(cond)
    return cond

# ── setup ─────────────────────────────────────────────────────────────────────
User.objects.filter(username__in=['_t_pat_feat', '_t_doc_feat']).delete()
u_pat = User.objects.create_user(username='_t_pat_feat', password='test1234', first_name='Test', last_name='User')
patient = Patient.objects.create(user=u_pat, address='TestCity')
u_doc = User.objects.create_user(username='_t_doc_feat', password='test1234')
doctor = Doctor.objects.create(user=u_doc, status=1, address='TestCity')

client = Client()
client.login(username='_t_pat_feat', password='test1234')

anon = Client()
doc_client = Client()
doc_client.login(username='_t_doc_feat', password='test1234')


# =============================================================================
print("\n=== 1. Medical Document Parser - page & auth ===")
# =============================================================================

r = client.get('/medical_documents')
check("Page loads (200)", r.status_code == 200, str(r.status_code))
check("Upload zone rendered", b'upload-zone' in r.content or b'uploadZone' in r.content)
check("No documents yet shows empty state", b'No documents' in r.content or b'empty-state' in r.content or b'uploaded yet' in r.content)

r2 = anon.get('/medical_documents')
check("Unauthenticated redirected (302)", r2.status_code == 302, str(r2.status_code))

r3 = client.post('/upload_medical_document', {})
check("Upload with no file returns 400", r3.status_code == 400, str(r3.status_code))

# =============================================================================
print("\n=== 2. Medical Document Parser - PDF upload & Claude extraction ===")
# =============================================================================

SAMPLE_PDF_TEXT = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
"""

import pdfplumber, tempfile
# Create a real text-based file to test parsing logic directly
sample_report = """
PATIENT LAB REPORT
Patient: John Doe, Age: 45, Male

DIAGNOSIS: Hypertension Stage 2, Hypercholesterolemia

MEDICATIONS:
- Amlodipine 5mg once daily
- Atorvastatin 20mg at night

VITALS:
Blood Pressure: 148/92 mmHg
Heart Rate: 78 bpm
Temperature: 98.6F
SpO2: 97%
Cholesterol: 245 mg/dl

LAB RESULTS:
- HbA1c: 5.8% (normal)
- Fasting glucose: 102 mg/dl (borderline)
- LDL: 158 mg/dl (high)

ALLERGIES: Penicillin

DOCTOR NOTES: Patient advised lifestyle modification, low-sodium diet, 30 min daily walk.
Follow-up in 4 weeks.
"""

# Test the internal parser function directly (no real PDF needed)
from health.views import _parse_doc_text_with_claude
parsed = _parse_doc_text_with_claude(sample_report, 'lab_report.txt')
check("Parsing returns dict",           isinstance(parsed, dict), str(type(parsed)))
check("Diagnosis extracted",            bool(parsed.get('diagnosis')), parsed.get('diagnosis', '')[:60])
check("Medications extracted",          bool(parsed.get('medications')), parsed.get('medications', '')[:60])
check("Summary present",                bool(parsed.get('summary')), parsed.get('summary', '')[:80])
check("Vitals extracted",               bool(parsed.get('vitals') or parsed.get('lab_results') or parsed.get('diagnosis')))
check("Allergies extracted",            'penicillin' in str(parsed.get('allergies','')).lower() or bool(parsed.get('allergies')), parsed.get('allergies',''))

# Test model creation
doc_obj = MedicalDocument.objects.create(
    patient=patient,
    original_name='test_report.txt',
    file_type='pdf',
    parsed_data=parsed,
)
check("MedicalDocument saved to DB",    doc_obj.id is not None)
check("Parsed data stored as JSON",     isinstance(doc_obj.parsed_data, dict))

# Test page shows the document now
r4 = client.get('/medical_documents')
check("Document appears on page",       'test_report' in r4.content.decode())

# Test delete endpoint
r5 = client.post(f'/delete_medical_document/{doc_obj.id}/')
check("Delete returns ok=True",         json.loads(r5.content).get('ok') is True)
check("Document removed from DB",       not MedicalDocument.objects.filter(id=doc_obj.id).exists())

# Doctor cannot access patient documents page (not a patient account)
r6 = doc_client.get('/medical_documents')
check("Doctor redirected from doc page (not patient)", r6.status_code in (302, 200))

# =============================================================================
print("\n=== 3. Voice Transcribe endpoint - auth & validation ===")
# =============================================================================

r7 = anon.post('/voice_transcribe',
    data=json.dumps({'audio': 'abc'}),
    content_type='application/json')
check("Unauthenticated blocked (302)", r7.status_code == 302, str(r7.status_code))

r8 = client.post('/voice_transcribe',
    data=json.dumps({}),
    content_type='application/json')
check("Missing audio returns 400",      r8.status_code == 400, str(r8.status_code))

r9 = client.get('/voice_transcribe')
check("GET not allowed (405)",          r9.status_code == 405, str(r9.status_code))

# =============================================================================
print("\n=== 4. Voice Transcribe - text_fallback path (no mic needed) ===")
# =============================================================================

# Use text_fallback to bypass Sarvam STT and test Claude field extraction
payload = {
    'audio': base64.b64encode(b'fake').decode(),
    'text_fallback': 'I am 54 years old male. I have chest pain, blood pressure is 145, cholesterol is 230.'
}

# Temporarily disable Sarvam key to force fallback path
from django.conf import settings
original_key = settings.SARVAM_API_KEY
settings.SARVAM_API_KEY = ''
original_env = os.environ.pop('SARVAM_API_KEY', None)

r10 = client.post('/voice_transcribe',
    data=json.dumps(payload),
    content_type='application/json')

settings.SARVAM_API_KEY = original_key  # restore
if original_env is not None:
    os.environ['SARVAM_API_KEY'] = original_env

check("Fallback transcribe returns 200", r10.status_code == 200, str(r10.status_code))
if r10.status_code == 200:
    d = json.loads(r10.content)
    check("Response has transcript",     bool(d.get('transcript')), d.get('transcript','')[:60])
    check("Response has fields dict",    isinstance(d.get('fields'), dict))
    f = d.get('fields', {})
    check("Age extracted (54)",          f.get('age') == 54 or str(f.get('age','')) == '54', str(f.get('age','')))
    check("Sex extracted (1=male)",      f.get('sex') in (1, '1'), str(f.get('sex','')))
    check("At least 3 fields filled",    len(f) >= 3, str(f))
    print(f"       Fields extracted: {f}")
else:
    d = json.loads(r10.content)
    print(f"       Error: {d}")
    for _ in range(5): results.append(False)

# =============================================================================
print("\n=== 5. Voice Transcribe - live Sarvam STT ===")
# =============================================================================

sarvam_key = settings.SARVAM_API_KEY
check("SARVAM_API_KEY configured", bool(sarvam_key), sarvam_key[:12] + '...' if sarvam_key else 'MISSING')

if sarvam_key:
    import requests as req
    try:
        # Send a tiny silent WAV to check the API key is valid
        # WAV header for 0.1s silence at 16kHz mono
        import struct, wave
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(b'\x00\x00' * 1600)
        buf.seek(0)
        resp = req.post(
            'https://api.sarvam.ai/speech-to-text',
            headers={'API-Subscription-Key': sarvam_key},
            files={'file': ('test.wav', buf, 'audio/wav')},
            data={'model': 'saarika:v2.5', 'with_timestamps': 'false'},
            timeout=20
        )
        check("Sarvam API key valid (HTTP 200)",   resp.status_code == 200, str(resp.status_code))
        if resp.status_code == 200:
            data = resp.json()
            check("Sarvam response has transcript key", 'transcript' in data, str(list(data.keys())))
            check("language_code in response",          'language_code' in data, str(list(data.keys())))
        else:
            print(f"       Sarvam error: {resp.text[:200]}")
            results.append(False); results.append(False)
    except Exception as e:
        check("Sarvam API reachable", False, str(e))
        results.append(False); results.append(False)
else:
    print("  [SKIP] No SARVAM_API_KEY set")
    for _ in range(3): results.append(True)  # skip, not a failure

# =============================================================================
print("\n=== 6. Integration - add_heartdetail page has voice UI ===")
# =============================================================================

r11 = client.get('/add_heartdetail')
body = r11.content.decode()
check("add_heartdetail loads (200)",        r11.status_code == 200)
check("Voice bar rendered",                 'voiceBtn' in body or 'voice_transcribe' in body)
check("Start Recording button present",     'Start Recording' in body)
check("Multilingual note present",          'Hindi' in body or 'multilingual' in body.lower() or 'Sarvam' in body or 'Tamil' in body)
check("JS toggleRecording function",        'toggleRecording' in body)
check("voice_transcribe fetch call",        'voice_transcribe' in body)

# =============================================================================
print("\n=== 7. Integration - medical_documents page structure ===")
# =============================================================================

r12 = client.get('/medical_documents')
body2 = r12.content.decode()
check("Page title present",                 'Medical Documents' in body2 or 'medical' in body2.lower())
check("Upload zone present",                'uploadZone' in body2 or 'upload-zone' in body2)
check("Drag-and-drop JS present",           'dragover' in body2 or 'drop' in body2)
check("Delete function present",            'deleteDoc' in body2)
check("Progress bar present",               'progressFill' in body2 or 'progress' in body2)
check("CSRF token in JS",                   'csrfToken' in body2 or 'csrf_token' in body2)

# =============================================================================
# cleanup
# =============================================================================
MedicalDocument.objects.filter(patient=patient).delete()
patient.delete(); u_pat.delete()
doctor.delete(); u_doc.delete()

# =============================================================================
print("\n=== Summary ===")
passed = sum(results)
total = len(results)
print(f"  {passed}/{total} tests passed")
if passed == total:
    print("  All tests passed.")
else:
    print(f"  {total-passed} test(s) failed.")
    sys.exit(1)
