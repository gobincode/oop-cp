import django, os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'health_desease.settings')
django.setup()

from django.core.cache import cache
from health.models import AIBookedAppointment, Patient
import datetime as _dt

call_sid = 'TEST_SID_001'

# Simulate cache populated by initiate_appointment_call
cache.set('call_data_' + call_sid, {
    'patient_contact': '8087980346',
    'patient_name': 'Vik Star',
    'hospital_name': 'Sardar Patel Cantonment General Hospital',
    'hospital_phone': '+918087980346',
    'hospital_address': 'Pune Cantonment',
    'reason': 'Cardiac consultation - Recent ECG analysis showed concerning results',
}, timeout=3600)

# Simulate cache populated by _extract_and_cache_booking
cache.set('call_booking_' + call_sid, {
    'date_str': 'tomorrow',
    'time_str': '9:00 PM',
    'doctor_name': 'Dr. Manu',
    'department': 'Cardiac Department',
}, timeout=3600)

print('Cache set. Simulating call_status completed handler...')

call_data = cache.get('call_data_' + call_sid, {})
booking   = cache.get('call_booking_' + call_sid, {})

patient_contact = call_data.get('patient_contact', '')
last10 = patient_contact.replace('+', '')[-10:]
patient_obj = Patient.objects.filter(contact__icontains=last10).first()
print(f'Patient lookup (last10={last10}): {patient_obj}')

if not patient_obj:
    print('No patient found - check that a patient with contact containing 8087980346 exists')
    all_patients = Patient.objects.all()[:5]
    for p in all_patients:
        print(f'  Existing patient: {p.user.username} contact={p.contact}')
    sys.exit(1)

# Parse date
date_str = booking.get('date_str', '')
appt_date = None
if 'tomorrow' in date_str.lower():
    appt_date = _dt.date.today() + _dt.timedelta(days=1)

# Parse time
time_str = booking.get('time_str', '')
appt_time = None
try:
    from dateutil import parser as _dp
    appt_time = _dp.parse(time_str, fuzzy=True).time()
    print(f'Parsed time: {appt_time}')
except Exception as e:
    print(f'Time parse error: {e}')

# Clean up previous test entry
AIBookedAppointment.objects.filter(call_sid=call_sid).delete()

obj = AIBookedAppointment.objects.create(
    patient=patient_obj,
    hospital_name=call_data.get('hospital_name', ''),
    hospital_phone=call_data.get('hospital_phone', ''),
    hospital_address=call_data.get('hospital_address', ''),
    doctor_name=booking.get('doctor_name', ''),
    department=booking.get('department', ''),
    appointment_date=appt_date,
    appointment_time=appt_time,
    appointment_datetime_str=f"{date_str} {time_str}".strip(),
    reason=call_data.get('reason', ''),
    call_sid=call_sid,
    status='confirmed',
)

print(f'\nSUCCESS - Created AIBookedAppointment:')
print(f'  Patient  : {obj.patient.user.username}')
print(f'  Hospital : {obj.hospital_name}')
print(f'  Date     : {obj.appointment_date}')
print(f'  Time     : {obj.appointment_time}')
print(f'  Doctor   : {obj.doctor_name}')
print(f'  Dept     : {obj.department}')
print(f'  Call SID : {obj.call_sid}')
print(f'\nNow check /my_appointments/ - the record should appear there.')
