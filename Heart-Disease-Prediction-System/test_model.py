import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'health_desease.settings')

import django
django.setup()

from health.views import prdict_heart_disease

test_cases = [
    # (label, age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
    ("HEALTHY - young male, typical angina, great vitals",   [29, 1, 2, 120, 180, 0, 2, 210, 0, 0.0, 2, 0, 2]),
    ("HEALTHY - young female, normal chest, good HR",        [41, 0, 1, 130, 204, 0, 2, 172, 0, 0.0, 2, 0, 2]),
    ("HEALTHY - middle-aged, no symptoms",                   [45, 0, 0, 128, 200, 0, 0, 165, 0, 0.0, 2, 0, 2]),
    ("AT RISK - elderly male, high BP, pain",                [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]),
    ("AT RISK - old female, high chol, low HR",              [67, 0, 2, 160, 286, 0, 2, 108, 1, 1.5, 1, 3, 2]),
    ("AT RISK - exercise angina, ST depression",             [57, 1, 0, 150, 276, 0, 2, 112, 1, 3.5, 0, 1, 1]),
    ("BORDERLINE - middle-aged male",                        [54, 1, 1, 122, 286, 0, 0, 116, 1, 3.2, 1, 2, 2]),
]

print(f"\n{'='*70}")
print(f"{'LABEL':<45} {'PRED':>6}  {'RESULT':<12}  {'ACCURACY':>8}")
print(f"{'='*70}")
for label, vals in test_cases:
    acc, pred = prdict_heart_disease(vals)
    result = "HEALTHY" if pred[0] == 0 else "AT RISK"
    print(f"{label:<45} {pred[0]:>6}  {result:<12}  {acc:>7.2f}%")
print(f"{'='*70}\n")
