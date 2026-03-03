"""
Simulates exactly what happens when a user submits the add_heartdetail form.
Mirrors the view's POST parsing and prdict_heart_disease call.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'health_desease.settings')
import django
django.setup()

from health.views import prdict_heart_disease

def simulate_form_post(form_data: dict):
    """
    Replicates the exact parsing logic in add_heartdetail view.
    form_data = dict of field: value (as strings, like a real POST).
    """
    # Prepend a fake csrfmiddlewaretoken to match the view's skip-first-key logic
    post = {'csrfmiddlewaretoken': ['fake'], **{k: [str(v)] for k, v in form_data.items()}}

    list_data = []
    count = 0
    for key, value in post.items():
        if count == 0:
            count = 1
            continue
        if key == "sex":
            raw = value[0].strip().lower()
            list_data.append(1 if raw in ('1', 'male', 'm') else 0)
            continue
        try:
            numeric_value = float(value[0])
            list_data.append(int(numeric_value) if numeric_value.is_integer() else numeric_value)
        except (ValueError, TypeError):
            list_data.append(value[0])

    acc, pred = prdict_heart_disease(list_data)
    result = "HEALTHY" if pred[0] == 0 else "AT RISK"
    return list_data, acc, pred[0], result


# Test cases: (label, form_fields)
# sex values match exactly what the radio buttons POST: "1"=male, "0"=female
cases = [
    ("Healthy young male",
     dict(age=29, sex="1", cp=2, trestbps=120, chol=180, fbs=0,
          restecg=2, thalach=210, exang=0, oldpeak=0.0, slope=2, ca=0, thal=2)),

    ("Healthy young female",
     dict(age=35, sex="0", cp=2, trestbps=120, chol=185, fbs=0,
          restecg=0, thalach=190, exang=0, oldpeak=0.0, slope=2, ca=0, thal=2)),

    ("At risk - elderly male, blocked vessels",
     dict(age=63, sex="1", cp=0, trestbps=145, chol=233, fbs=1,
          restecg=0, thalach=150, exang=0, oldpeak=2.3, slope=0, ca=2, thal=3)),

    ("At risk - old female, exercise angina",
     dict(age=67, sex="0", cp=0, trestbps=160, chol=286, fbs=0,
          restecg=0, thalach=108, exang=1, oldpeak=1.5, slope=1, ca=3, thal=3)),

    ("At risk - exang + ST depression",
     dict(age=57, sex="1", cp=0, trestbps=150, chol=276, fbs=0,
          restecg=0, thalach=112, exang=1, oldpeak=3.5, slope=0, ca=2, thal=3)),

    ("Borderline middle-aged",
     dict(age=54, sex="1", cp=1, trestbps=130, chol=250, fbs=0,
          restecg=0, thalach=140, exang=0, oldpeak=1.0, slope=1, ca=1, thal=2)),
]

print(f"\n{'='*75}")
print(f"{'LABEL':<42} {'PARSED LIST':>4}  {'RESULT':<10}  {'ACC':>7}")
print(f"{'='*75}")
for label, form in cases:
    parsed, acc, pred_val, result = simulate_form_post(form)
    short = str(parsed)[:28] + '..'
    print(f"{label:<42} {short:<30}  {result:<10}  {acc:>6.1f}%")
print(f"{'='*75}")

# Verify models are loaded from disk (not retrained)
print("\n--- Verifying pre-trained models are used (not retrained) ---")
import time
start = time.time()
simulate_form_post(cases[0][1])
elapsed = time.time() - start
if elapsed < 2.0:
    print(f"Prediction took {elapsed:.2f}s — pre-trained models loaded (fast). OK")
else:
    print(f"Prediction took {elapsed:.2f}s — WARNING: may be retraining (slow)")
