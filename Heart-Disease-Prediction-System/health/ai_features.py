"""
AI-powered clinical features using the Claude API.
"""
import ast
import os
from datetime import date
from django.conf import settings


def _call(system_prompt, user_message, max_tokens=300):
    from .ai_calling_agent import call_claude
    api_key = getattr(settings, 'CLAUDE_API_KEY', '') or os.getenv('CLAUDE_API_KEY', '')
    if not api_key:
        raise ValueError('CLAUDE_API_KEY not configured')
    return call_claude(system_prompt, user_message, api_key, max_tokens=max_tokens)


# ─── Feature 1: AI Doctor Recommendations ────────────────────────────────────

FEATURE_LABELS = [
    'Age', 'Sex (1=M, 0=F)', 'Chest Pain Type (0–3)',
    'Resting BP (mmHg)', 'Cholesterol (mg/dl)',
    'Fasting Blood Sugar >120 mg/dl', 'Resting ECG (0–2)',
    'Max Heart Rate Achieved', 'Exercise-Induced Angina (1=yes)',
    'ST Depression (oldpeak)', 'ST Slope (0–2)',
    'Major Vessels (0–3)', 'Thalassemia (0–3)',
]

_RECOMMENDATION_SYSTEM = """You are a medical triage AI. Given patient vitals and an ML prediction result, recommend a specialist and urgency level.
Reply in EXACTLY this format with no extra text:
SPECIALTY: <single specialty name>
URGENCY: <low|medium|high>
REASON: <one plain sentence>
ALSO_SEE: <comma-separated 1-2 other specialties, or None>"""


def get_doctor_recommendation(pred, values_list_str, patient_name='Patient'):
    """Return a recommendation dict for the predict_disease template."""
    try:
        values = ast.literal_eval(str(values_list_str))
        symptom_lines = '\n'.join(
            f'  {FEATURE_LABELS[i]}: {values[i]}'
            for i in range(min(len(values), len(FEATURE_LABELS)))
        )
    except Exception:
        symptom_lines = str(values_list_str)

    result_text = 'AT RISK for heart disease' if str(pred) != '0' else 'HEALTHY (no heart disease detected)'

    user_msg = f"Patient: {patient_name}\nML Prediction: {result_text}\nVitals:\n{symptom_lines}\n\nGive your recommendation."

    try:
        response = _call(_RECOMMENDATION_SYSTEM, user_msg, max_tokens=150)
        return _parse_recommendation(response)
    except Exception:
        return _default_recommendation(pred)


def _parse_recommendation(text):
    lines = {}
    for line in text.strip().splitlines():
        if ':' in line:
            key, _, val = line.partition(':')
            lines[key.strip()] = val.strip()

    urgency = lines.get('URGENCY', 'medium').lower()
    urgency_map = {
        'high':   ('high',   'Urgent — seek care within 24–48 hours', '#8b2e2e', '#fdf0ee', '#e8b4a8'),
        'medium': ('medium', 'Soon — schedule within 1–2 weeks',      '#7a5a1a', '#fef8ec', '#e8d9a0'),
        'low':    ('low',    'Routine — book a check-up',              '#4a6b4d', '#eaf0ea', '#b8d4ba'),
    }
    u = urgency_map.get(urgency, urgency_map['medium'])

    also_raw = lines.get('ALSO_SEE', 'None')
    also = [s.strip() for s in also_raw.split(',') if s.strip().lower() not in ('', 'none')]

    return {
        'specialty':      lines.get('SPECIALTY', 'Cardiologist'),
        'urgency':        u[0],
        'urgency_label':  u[1],
        'urgency_color':  u[2],
        'urgency_bg':     u[3],
        'urgency_border': u[4],
        'reason':         lines.get('REASON', 'Please consult a specialist based on your results.'),
        'also_see':       also,
    }


def _default_recommendation(pred):
    if str(pred) != '0':
        return {
            'specialty': 'Cardiologist',
            'urgency': 'high',
            'urgency_label': 'Urgent — seek care within 24–48 hours',
            'urgency_color': '#8b2e2e',
            'urgency_bg': '#fdf0ee',
            'urgency_border': '#e8b4a8',
            'reason': 'Heart disease risk detected. A cardiologist should perform a thorough evaluation.',
            'also_see': ['General Physician'],
        }
    return {
        'specialty': 'General Physician',
        'urgency': 'low',
        'urgency_label': 'Routine — annual check-up recommended',
        'urgency_color': '#4a6b4d',
        'urgency_bg': '#eaf0ea',
        'urgency_border': '#b8d4ba',
        'reason': 'No current risk detected. Maintain annual check-ups to stay healthy.',
        'also_see': [],
    }


# ─── Feature 2: Doctor AI Patient Summary ────────────────────────────────────

_SUMMARY_SYSTEM = """You are a clinical AI assistant helping a doctor quickly review a patient's health history.
Write a concise clinical summary (3–5 sentences) in plain conversational English.
Cover: overall risk level, key findings, any trend over time, and one suggested next step for the doctor.
No markdown, no bullets, no bold — just clear readable prose."""


def get_patient_summary(patient_obj):
    """Return a plain-text AI clinical summary for a patient."""
    from .models import Search_Data, ECG_Prediction, Appointment

    lines = [f"Patient: {patient_obj.user.get_full_name() or patient_obj.user.username}"]
    if patient_obj.dob:
        age = (date.today() - patient_obj.dob).days // 365
        lines.append(f"Age: {age}")

    predictions = Search_Data.objects.filter(patient=patient_obj).order_by('-id')[:5]
    if predictions:
        lines.append(f"\nHeart Disease Predictions ({predictions.count()} recent):")
        for p in predictions:
            label = 'AT RISK' if str(p.result) != '0' else 'HEALTHY'
            lines.append(f"  {p.created.strftime('%b %d %Y')}: {label} (accuracy {p.prediction_accuracy}%)")

    ecg_records = ECG_Prediction.objects.filter(patient=patient_obj).order_by('-created')[:3]
    if ecg_records:
        lines.append(f"\nECG Results ({ecg_records.count()} recent):")
        for e in ecg_records:
            conf = f'{e.confidence:.0%}' if e.confidence else 'N/A'
            lines.append(f"  {e.created.strftime('%b %d %Y')}: {e.prediction_label} (confidence {conf})")

    appointments = Appointment.objects.filter(patient=patient_obj).order_by('-appointment_date')[:5]
    if appointments:
        lines.append(f"\nAppointments ({appointments.count()} recent):")
        for a in appointments:
            lines.append(f"  {a.appointment_date.strftime('%b %d %Y')}: {a.status}")

    context = '\n'.join(lines)

    try:
        return _call(_SUMMARY_SYSTEM, f"Generate a clinical summary:\n\n{context}", max_tokens=350)
    except Exception as e:
        return f"Summary generation failed: {str(e)}"
