"""
Comprehensive tests for AI Doctor Recommendations & Doctor AI Patient Summary.
Run: python test_ai_features.py
"""
import os, sys, django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'health_desease.settings')
sys.path.insert(0, os.path.dirname(__file__))
django.setup()

from django.test import TestCase, Client
from django.contrib.auth.models import User
from health.models import Patient, Doctor, Search_Data, Appointment
from datetime import date, time, timedelta

# ── helpers ──────────────────────────────────────────────────────────────────
PASS = '\033[92m[PASS]\033[0m'
FAIL = '\033[91m[FAIL]\033[0m'

def check(label, cond, detail=''):
    status = PASS if cond else FAIL
    print(f"  {status} {label}" + (f"  →  {detail}" if detail else ''))
    return cond

results = []

# ─────────────────────────────────────────────────────────────────────────────
print("\n═══ 1. AI Doctor Recommendation – unit tests ═══")

from health.ai_features import get_doctor_recommendation, _parse_recommendation, _default_recommendation

# 1a. fallback for AT RISK
r = _default_recommendation('1')
results.append(check("At-risk default returns Cardiologist",      r['specialty'] == 'Cardiologist'))
results.append(check("At-risk default urgency = high",             r['urgency'] == 'high'))

# 1b. fallback for HEALTHY
r = _default_recommendation('0')
results.append(check("Healthy default returns General Physician",  r['specialty'] == 'General Physician'))
results.append(check("Healthy default urgency = low",              r['urgency'] == 'low'))

# 1c. _parse_recommendation with full text
sample = """SPECIALTY: Cardiologist
URGENCY: high
REASON: Elevated cholesterol and ST depression indicate cardiac risk.
ALSO_SEE: General Physician, Cardiac Surgeon"""
p = _parse_recommendation(sample)
results.append(check("Parse: specialty",           p['specialty'] == 'Cardiologist'))
results.append(check("Parse: urgency",             p['urgency']   == 'high'))
results.append(check("Parse: reason present",      bool(p['reason'])))
results.append(check("Parse: also_see has 2",      len(p['also_see']) == 2,  str(p['also_see'])))
results.append(check("Parse: urgency_color set",   bool(p['urgency_color'])))

# 1d. _parse_recommendation with ALSO_SEE = None
sample2 = "SPECIALTY: GP\nURGENCY: low\nREASON: Routine check.\nALSO_SEE: None"
p2 = _parse_recommendation(sample2)
results.append(check("Parse ALSO_SEE None → empty list", p2['also_see'] == []))

# 1e. get_doctor_recommendation with real vitals (at-risk profile)
at_risk = [62, 1, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2]
r = get_doctor_recommendation('1', str(at_risk), 'TestPatient')
results.append(check("Live recommendation has specialty",   bool(r.get('specialty')), r.get('specialty','')))
results.append(check("Live recommendation has urgency",     r.get('urgency') in ('low','medium','high'), r.get('urgency','')))
results.append(check("Live recommendation has reason",      bool(r.get('reason'))))
results.append(check("Live recommendation has urgency_bg",  bool(r.get('urgency_bg'))))

# 1f. get_doctor_recommendation with healthy profile
healthy = [52, 0, 2, 120, 196, 0, 1, 162, 0, 0.2, 2, 0, 2]
rh = get_doctor_recommendation('0', str(healthy), 'TestPatient')
results.append(check("Healthy rec has specialty",   bool(rh.get('specialty'))))
results.append(check("Healthy rec has reason",      bool(rh.get('reason'))))

# ─────────────────────────────────────────────────────────────────────────────
print("\n═══ 2. Doctor AI Patient Summary – unit tests ═══")

from health.ai_features import get_patient_summary

# Create temporary test users/patient
u_patient = User.objects.create_user(username='_test_summ_pt', password='x', first_name='Test', last_name='Patient')
patient_obj = Patient.objects.create(user=u_patient, dob=date(1985, 3, 15))

# Add some prediction records
Search_Data.objects.create(patient=patient_obj, prediction_accuracy='83.0', result='1',
                           values_list=str([62,1,0,140,268,0,0,160,0,3.6,0,2,2]))
Search_Data.objects.create(patient=patient_obj, prediction_accuracy='83.0', result='0',
                           values_list=str([52,0,2,120,196,0,1,162,0,0.2,2,0,2]))

summary = get_patient_summary(patient_obj)
results.append(check("Summary is non-empty string", isinstance(summary, str) and len(summary) > 30, summary[:80]))
results.append(check("Summary is not an error",     not summary.startswith('Summary generation failed')))
results.append(check("Summary > 50 chars",          len(summary) > 50))

# Cleanup
Search_Data.objects.filter(patient=patient_obj).delete()
patient_obj.delete()
u_patient.delete()

# ─────────────────────────────────────────────────────────────────────────────
print("\n═══ 3. HTTP endpoint tests ═══")

# Create test users
u_pat = User.objects.create_user(username='_test_ep_pat', password='testpass123')
pat   = Patient.objects.create(user=u_pat, address='Test City')

u_doc = User.objects.create_user(username='_test_ep_doc', password='testpass123')
doc   = Doctor.objects.create(user=u_doc, status=1, address='Test City')

# Prediction record
Search_Data.objects.create(patient=pat, prediction_accuracy='83.0', result='1',
                           values_list=str([62,1,0,140,268,0,0,160,0,3.6,0,2,2]))

client_pat = Client()
client_pat.login(username='_test_ep_pat', password='testpass123')

client_doc = Client()
client_doc.login(username='_test_ep_doc', password='testpass123')

# 3a. predict_desease renders OK with recommendation in context
resp = client_pat.get(f'/predict_desease/1/83.0/')
results.append(check("predict_desease 200 for at-risk",   resp.status_code == 200, str(resp.status_code)))
body = resp.content.decode()
results.append(check("Recommendation card rendered",       'AI Specialist Recommendation' in body))
results.append(check("Specialty text present",             'See a ' in body))
results.append(check("Urgency badge present",              'HIGH' in body or 'MEDIUM' in body or 'LOW' in body))

# 3b. predict_desease healthy
resp2 = client_pat.get('/predict_desease/0/83.0/')
results.append(check("predict_desease 200 for healthy",    resp2.status_code == 200))
body2 = resp2.content.decode()
results.append(check("Healthy recommendation rendered",    'AI Specialist Recommendation' in body2))

# 3c. ai_patient_summary – doctor can access
resp3 = client_doc.get(f'/ai_patient_summary/{pat.id}/')
results.append(check("Doctor can access summary endpoint", resp3.status_code == 200, str(resp3.status_code)))
import json
data3 = json.loads(resp3.content)
results.append(check("Summary JSON has 'summary' key",     'summary' in data3))
results.append(check("Summary text is non-empty",          len(data3.get('summary','')) > 20, data3.get('summary','')[:60]))

# 3d. ai_patient_summary – patient CANNOT access (403)
resp4 = client_pat.get(f'/ai_patient_summary/{pat.id}/')
results.append(check("Patient blocked from summary (403)", resp4.status_code == 403, str(resp4.status_code)))

# 3e. ai_patient_summary – unauthenticated redirected
c_anon = Client()
resp5 = c_anon.get(f'/ai_patient_summary/{pat.id}/')
results.append(check("Unauthenticated blocked (302/403)",  resp5.status_code in (302, 403), str(resp5.status_code)))

# 3f. ai_patient_summary – invalid patient id → 404
resp6 = client_doc.get('/ai_patient_summary/999999/')
results.append(check("Non-existent patient → 404",         resp6.status_code == 404, str(resp6.status_code)))

# 3g. doctor_home renders with AI Summary buttons
resp7 = client_doc.get('/doctor_home')
results.append(check("Doctor home 200",                    resp7.status_code == 200))
body7 = resp7.content.decode()
results.append(check("AI Summary button present in DOM",   'AI Summary' in body7 or 'loadSummary' in body7))
results.append(check("Summary modal present in DOM",       'summaryModal' in body7))

# Cleanup
Search_Data.objects.filter(patient=pat).delete()
pat.delete(); u_pat.delete()
doc.delete(); u_doc.delete()

# ─────────────────────────────────────────────────────────────────────────────
print("\n═══ Summary ═══")
passed = sum(results)
total  = len(results)
print(f"  {passed}/{total} tests passed")
if passed == total:
    print("  \033[92mAll tests passed.\033[0m")
else:
    print(f"  \033[91m{total - passed} test(s) failed.\033[0m")
    sys.exit(1)
