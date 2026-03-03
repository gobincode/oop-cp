"""Test both new views end-to-end using Django's test client."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'health_desease.settings')

import django
django.setup()

from django.test import RequestFactory, Client
from django.contrib.auth.models import User
from health.models import Patient

# Find the first patient user
patient_user = None
for p in Patient.objects.select_related('user').all():
    if p.user:
        patient_user = p.user
        break

if not patient_user:
    print("SKIP: no patient in DB")
    sys.exit(0)

print(f"Testing with patient: {patient_user.username}")

c = Client()
c.force_login(patient_user)

# 1. Trend dashboard HTML
r = c.get('/health_trends')
print(f"health_trends         -> HTTP {r.status_code}  ({'OK' if r.status_code == 200 else 'FAIL'})")

# 2. JSON data endpoint
r = c.get('/health_trends_data/')
print(f"health_trends_data    -> HTTP {r.status_code}  ({'OK' if r.status_code == 200 else 'FAIL'})")
if r.status_code == 200:
    import json
    data = json.loads(r.content)
    print(f"  predictions records : {len(data['predictions']['labels'])}")
    print(f"  ecg records         : {len(data['ecg']['labels'])}")

# 3. PDF download
r = c.get('/download_health_report')
print(f"download_health_report-> HTTP {r.status_code}  ({'OK' if r.status_code == 200 else 'FAIL'})")
if r.status_code == 200:
    ct = r.get('Content-Type', '')
    cd = r.get('Content-Disposition', '')
    print(f"  Content-Type        : {ct}")
    print(f"  Content-Disposition : {cd}")
    print(f"  PDF size            : {len(r.content):,} bytes")

print("\nAll tests done.")
