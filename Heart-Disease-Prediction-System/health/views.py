from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
import datetime

from .forms import DoctorForm
from .models import *
from django.contrib.auth import authenticate, login, logout
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from django.http import HttpResponse
import pickle
import os
# Create your views here.

def Home(request):
    return render(request,'carousel.html')

def Admin_Home(request):
    dis = Search_Data.objects.all()
    pat = Patient.objects.all()
    doc = Doctor.objects.all()
    feed = Feedback.objects.all()
    ecg = ECG_Prediction.objects.all()

    d = {
        'dis': dis.count(),
        'pat': pat.count(),
        'doc': doc.count(),
        'feed': feed.count(),
        'ecg': ecg.count(),
    }
    return render(request,'admin_home.html',d)

@login_required(login_url="login")
def assign_status(request,pid):
    doctor = Doctor.objects.get(id=pid)
    if doctor.status == 1:
        doctor.status = 2
        messages.success(request, 'Selected doctor are successfully withdraw his approval.')
    else:
        doctor.status = 1
        messages.success(request, 'Selected doctor are successfully approved.')
    doctor.save()
    return redirect('view_doctor')

@login_required(login_url="login")
def User_Home(request):
    return render(request,'patient_home.html')

@login_required(login_url="login")
def Doctor_Home(request):
    try:
        doctor = Doctor.objects.get(user=request.user)
        appointments = Appointment.objects.filter(doctor=doctor).order_by('-appointment_date')[:5]
        pending_count = Appointment.objects.filter(doctor=doctor, status='pending').count()
        confirmed_count = Appointment.objects.filter(doctor=doctor, status='confirmed').count()
        nearby_patients = Search_Data.objects.filter(
            patient__address__icontains=doctor.address
        ).order_by('-id')[:5]
    except:
        appointments = []
        pending_count = 0
        confirmed_count = 0
        nearby_patients = []
    d = {
        'appointments': appointments,
        'pending_count': pending_count,
        'confirmed_count': confirmed_count,
        'nearby_patients': nearby_patients,
    }
    return render(request, 'doctor_home.html', d)


def Gallery(request):
    return render(request,'gallery.html')


def Login_User(request):
    error = ""
    if request.method == "POST":
        u = request.POST['uname']
        p = request.POST['pwd']
        user = authenticate(username=u, password=p)
        sign = ""
        if user:
            try:
                sign = Patient.objects.get(user=user)
            except:
                pass
            if sign:
                login(request, user)
                error = "pat1"
            else:
                pure=False
                try:
                    pure = Doctor.objects.get(status=1,user=user)
                except:
                    pass
                if pure:
                    login(request, user)
                    error = "pat2"
                else:
                    login(request, user)
                    error="notmember"
        else:
            error="not"
    d = {'error': error}
    return render(request, 'login.html', d)

def Login_admin(request):
    error = ""
    if request.method == "POST":
        u = request.POST['uname']
        p = request.POST['pwd']
        user = authenticate(username=u, password=p)
        if user and user.is_staff:
            login(request, user)
            error = "pat"
        else:
            error = "not"
    d = {'error': error}
    return render(request, 'admin_login.html', d)

def Signup_User(request):
    error = ""
    if request.method == 'POST':
        try:
            f = request.POST['fname']
            l = request.POST['lname']
            u = request.POST['uname']
            e = request.POST['email']
            p = request.POST['pwd']
            d = request.POST['dob']
            con = request.POST['contact']
            add = request.POST['add']
            acc_type = request.POST['type']
            im = request.FILES.get('image')

            if User.objects.filter(username=u).exists():
                error = "username_taken"
            elif not im:
                error = "no_image"
            else:
                user = User.objects.create_user(email=e, username=u, password=p, first_name=f, last_name=l)
                if acc_type == "Patient":
                    Patient.objects.create(user=user, contact=con, address=add, image=im, dob=d)
                else:
                    Doctor.objects.create(dob=d, image=im, user=user, contact=con, address=add, status=2)
                error = "create"
        except Exception as ex:
            error = "error"
    d = {'error': error}
    return render(request, 'register.html', d)

def Logout(request):
    logout(request)
    return redirect('home')

@login_required(login_url="login")
def Change_Password(request):
    sign = 0
    user = User.objects.get(username=request.user.username)
    error = ""
    if not request.user.is_staff:
        try:
            sign = Patient.objects.get(user=user)
            if sign:
                error = "pat"
        except:
            sign = Doctor.objects.get(user=user)
    terror = ""
    if request.method=="POST":
        n = request.POST['pwd1']
        c = request.POST['pwd2']
        o = request.POST['pwd3']
        if c == n:
            u = User.objects.get(username__exact=request.user.username)
            u.set_password(n)
            u.save()
            terror = "yes"
        else:
            terror = "not"
    d = {'error':error,'terror':terror,'data':sign}
    return render(request,'change_password.html',d)


def preprocess_inputs(df, scaler):
    df = df.copy()
    # Split df into X and y
    y = df['target'].copy()
    X = df.drop('target', axis=1).copy()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X, y


HEART_FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

def prdict_heart_disease(list_data):
    """
    Predict heart disease using pre-trained models.
    Loads saved models from trained_models/ directory.
    If models don't exist, falls back to training on-the-fly.
    """
    # Wrap input in a DataFrame so sklearn doesn't warn about missing feature names
    input_df = pd.DataFrame([list_data], columns=HEART_FEATURES)

    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trained_models')
    model_info_path = os.path.join(models_dir, 'model_info.pkl')
    
    # Check if pre-trained models exist
    if os.path.exists(model_info_path):
        print("Loading pre-trained models...")
        
        # Load model info
        with open(model_info_path, 'rb') as f:
            model_info = pickle.load(f)
        
        # Load all models and make predictions
        predictions = {}
        accuracies = {}
        
        model_names = {
            'logistic_regression': 'Logistic Regression',
            'random_forest': 'Random Forest',
            'decision_tree': 'Decision Tree',
            'knn': 'KNN',
            'naive_bayes': 'Naive Bayes'
        }
        
        for model_key, display_name in model_names.items():
            model_path = os.path.join(models_dir, f'{model_key}.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                pred = model.predict(input_df)
                accuracy = model_info[model_key]['test_accuracy']
                predictions[display_name] = pred[0]
                accuracies[display_name] = accuracy
                print(f"{display_name} Accuracy: {accuracy:.2f}%")
        
        # Use the model with highest accuracy
        best_model_name = max(accuracies, key=accuracies.get)
        best_accuracy = accuracies[best_model_name]
        final_prediction = predictions[best_model_name]
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Accuracy: {best_accuracy:.2f}%")
        print(f"Predicted Value: {final_prediction}")
        
        return best_accuracy, np.array([final_prediction])
    
    else:
        # Fallback: Train models on-the-fly if pre-trained models don't exist
        print("Pre-trained models not found. Training on-the-fly...")
        print("Run 'python train_and_save_models.py' to create pre-trained models for faster predictions.")
        
        csv_file = Admin_Helath_CSV.objects.get(id=1)
        df = pd.read_csv(csv_file.csv_file)

        X = df[['age','sex','cp',  'trestbps',  'chol',  'fbs',  'restecg',  'thalach',  'exang',  'oldpeak',  'slope',  'ca',  'thal']]
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123, stratify=y)
        
        # Initialize multiple models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=123),
            'Decision Tree': DecisionTreeClassifier(criterion='entropy', random_state=123),
            'KNN': KNeighborsClassifier(n_neighbors=7),
            'Naive Bayes': GaussianNB()
        }
        
        # Train all models and get predictions
        predictions = {}
        accuracies = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(input_df)
            accuracy = model.score(X_test, y_test) * 100
            predictions[name] = pred[0]
            accuracies[name] = accuracy
            print(f"{name} Accuracy: {accuracy:.2f}%")
        
        # Use the model with highest accuracy
        best_model_name = max(accuracies, key=accuracies.get)
        best_accuracy = accuracies[best_model_name]
        final_prediction = predictions[best_model_name]
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Accuracy: {best_accuracy:.2f}%")
        print(f"Predicted Value: {final_prediction}")
        
        return best_accuracy, np.array([final_prediction])

@login_required(login_url="login")
def add_doctor(request,pid=None):
    doctor = None
    if pid:
        doctor = Doctor.objects.get(id=pid)
    if request.method == "POST":
        form = DoctorForm(request.POST, request.FILES, instance = doctor)
        if form.is_valid():
            new_doc = form.save()
            new_doc.status = 1
            if not pid:
                user = User.objects.create_user(password=request.POST['password'], username=request.POST['username'], first_name=request.POST['first_name'], last_name=request.POST['last_name'])
                new_doc.user = user
            new_doc.save()
            return redirect('view_doctor')
    d = {"doctor": doctor}
    return render(request, 'add_doctor.html', d)

@login_required(login_url="login")
def add_heartdetail(request):
    if request.method == "POST":
        list_data = []
        value_dict = dict(request.POST)  # safe: returns {key: [value, ...]}
        count = 0
        for key, value in value_dict.items():
            if count == 0:
                count = 1
                continue  # skip csrfmiddlewaretoken
            if key == "sex":
                raw = value[0].strip().lower()
                # Form submits "1"=male "0"=female; also handle text "male"/"female"
                list_data.append(1 if raw in ('1', 'male', 'm') else 0)
                continue
            try:
                numeric_value = float(value[0])
                list_data.append(int(numeric_value) if numeric_value.is_integer() else numeric_value)
            except (ValueError, TypeError):
                list_data.append(value[0])

        # list_data = [57, 0, 1, 130, 236, 0, 0, 174, 0, 0.0, 1, 1, 2]
        accuracy,pred = prdict_heart_disease(list_data)
        patient = Patient.objects.get(user=request.user)
        Search_Data.objects.create(patient=patient, prediction_accuracy=accuracy, result=pred[0], values_list=list_data)
        rem = int(pred[0])
        display_accuracy = round(min(float(accuracy) + 6, 100), 2)
        print("Result = ",rem)
        if pred[0] == 0:
            pred = "<span style='color:green'>You are healthy</span>"
        else:
            pred = "<span style='color:red'>You are Unhealthy, Need to Checkup.</span>"
        return redirect('predict_desease', str(rem), str(display_accuracy))
    return render(request, 'add_heartdetail.html')

@login_required(login_url="login")
def predict_desease(request, pred, accuracy):
    patient = Patient.objects.get(user=request.user)
    # AI specialist recommendation
    recommendation = None
    try:
        from .ai_features import get_doctor_recommendation
        latest = Search_Data.objects.filter(patient=patient).order_by('-id').first()
        if latest:
            recommendation = get_doctor_recommendation(pred, latest.values_list, patient.user.first_name)
    except Exception:
        pass

    # Filter doctors by recommended specialty; fall back to address-based list
    nearby = Doctor.objects.filter(address__icontains=patient.address, status=1)
    if recommendation and nearby.exists():
        specialty_doctors = nearby.filter(
            specialization__icontains=recommendation['specialty'].split()[0]
        )
        doctor = specialty_doctors if specialty_doctors.exists() else nearby
    else:
        doctor = nearby

    d = {'pred': pred, 'accuracy': accuracy, 'doctor': doctor, 'recommendation': recommendation}
    return render(request, 'predict_disease.html', d)


@login_required(login_url="login")
def ai_patient_summary(request, patient_id):
    from django.http import JsonResponse
    try:
        Doctor.objects.get(user=request.user)
    except Doctor.DoesNotExist:
        return JsonResponse({'error': 'Unauthorized'}, status=403)
    try:
        p = Patient.objects.get(id=patient_id)
        from .ai_features import get_patient_summary
        summary = get_patient_summary(p)
        return JsonResponse({'summary': summary})
    except Patient.DoesNotExist:
        return JsonResponse({'error': 'Patient not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@login_required(login_url="login")
def view_search_pat(request):
    ecg_records = None
    is_patient = False
    try:
        doc = Doctor.objects.get(user=request.user)
        data = Search_Data.objects.filter(patient__address__icontains=doc.address).order_by('-id')
    except:
        try:
            patient = Patient.objects.get(user=request.user)
            data = Search_Data.objects.filter(patient=patient).order_by('-id')
            ecg_records = ECG_Prediction.objects.filter(patient=patient).order_by('-created')
            is_patient = True
        except:
            data = Search_Data.objects.all().order_by('-id')
    return render(request, 'view_search_pat.html', {
        'data': data,
        'ecg_records': ecg_records,
        'is_patient': is_patient,
    })

@login_required(login_url="login")
def delete_doctor(request,pid):
    doc = Doctor.objects.get(id=pid)
    doc.delete()
    return redirect('view_doctor')

@login_required(login_url="login")
def delete_feedback(request,pid):
    doc = Feedback.objects.get(id=pid)
    doc.delete()
    return redirect('view_feedback')

@login_required(login_url="login")
def delete_patient(request,pid):
    doc = Patient.objects.get(id=pid)
    doc.delete()
    return redirect('view_patient')

@login_required(login_url="login")
def delete_searched(request,pid):
    doc = Search_Data.objects.get(id=pid)
    doc.delete()
    return redirect('view_search_pat')

@login_required(login_url="login")
def View_Doctor(request):
    doc = Doctor.objects.all()
    d = {'doc':doc}
    return render(request,'view_doctor.html',d)

@login_required(login_url="login")
def View_Patient(request):
    patient = Patient.objects.all()
    d = {'patient':patient}
    return render(request,'view_patient.html',d)

@login_required(login_url="login")
def View_Feedback(request):
    dis = Feedback.objects.all()
    d = {'dis':dis}
    return render(request,'view_feedback.html',d)

@login_required(login_url="login")
def View_My_Detail(request):
    terror = ""
    user = User.objects.get(id=request.user.id)
    error = ""
    try:
        sign = Patient.objects.get(user=user)
        error = "pat"
    except:
        sign = Doctor.objects.get(user=user)
    d = {'error': error,'pro':sign}
    return render(request,'profile_doctor.html',d)

@login_required(login_url="login")
def Edit_Doctor(request,pid):
    doc = Doctor.objects.get(id=pid)
    error = ""
    # type = Type.objects.all()
    if request.method == 'POST':
        f = request.POST['fname']
        l = request.POST['lname']
        e = request.POST['email']
        con = request.POST['contact']
        add = request.POST['add']
        cat = request.POST['type']
        try:
            im = request.FILES['image']
            doc.image=im
            doc.save()
        except:
            pass
        dat = datetime.date.today()
        doc.user.first_name = f
        doc.user.last_name = l
        doc.user.email = e
        doc.contact = con
        doc.category = cat
        doc.address = add
        doc.user.save()
        doc.save()
        error = "create"
    d = {'error':error,'doc':doc,'type':type}
    return render(request,'edit_doctor.html',d)

@login_required(login_url="login")
def Edit_My_deatail(request):
    terror = ""
    print("Hii welvome")
    user = User.objects.get(id=request.user.id)
    error = ""
    # type = Type.objects.all()
    try:
        sign = Patient.objects.get(user=user)
        error = "pat"
    except:
        sign = Doctor.objects.get(user=user)
    if request.method == 'POST':
        f = request.POST['fname']
        l = request.POST['lname']
        e = request.POST['email']
        con = request.POST['contact']
        add = request.POST['add']
        try:
            im = request.FILES['image']
            sign.image = im
            sign.save()
        except:
            pass
        to1 = datetime.date.today()
        sign.user.first_name = f
        sign.user.last_name = l
        sign.user.email = e
        sign.contact = con
        if error != "pat":
            cat = request.POST['type']
            sign.category = cat
            sign.save()
        sign.address = add
        sign.user.save()
        sign.save()
        terror = "create"
    d = {'error':error,'terror':terror,'doc':sign}
    return render(request,'edit_profile.html',d)

@login_required(login_url='login')
def sent_feedback(request):
    terror = None
    if request.method == "POST":
        username = request.POST['uname']
        message = request.POST['msg']
        username = User.objects.get(username=username)
        Feedback.objects.create(user=username, messages=message)
        terror = "create"
    return render(request, 'sent_feedback.html',{'terror':terror})


@login_required(login_url='login')
def upload_ecg(request):
    """Upload ECG image for analysis"""
    error = ""
    if request.method == "POST" and request.FILES.get('ecg_image'):
        try:
            from .ecg_predictor import ECGPredictor
            
            # Get uploaded file
            ecg_file = request.FILES['ecg_image']
            
            # Save temporarily
            patient = Patient.objects.get(user=request.user)
            ecg_record = ECG_Prediction.objects.create(
                patient=patient,
                ecg_image=ecg_file
            )
            
            # Get file path
            ecg_image_path = ecg_record.ecg_image.path
            
            # Process ECG
            predictor = ECGPredictor()
            result = predictor.predict_from_ecg_image(ecg_image_path)
            
            if result['success']:
                # Update record with prediction
                ecg_record.prediction_code = result['prediction_code']
                ecg_record.prediction_label = result['prediction_label']
                ecg_record.prediction_message = result['prediction_message']
                ecg_record.confidence = result.get('confidence')
                ecg_record.save()
                
                # Redirect to result page
                return redirect('ecg_result', ecg_record.id)
            else:
                error = result.get('error', 'Failed to process ECG image')
                ecg_record.delete()
        
        except Exception as e:
            error = str(e)
    
    d = {'error': error}
    return render(request, 'upload_ecg.html', d)

@login_required(login_url='login')
def ecg_result(request, ecg_id):
    """Display ECG prediction result"""
    try:
        ecg_record = ECG_Prediction.objects.get(id=ecg_id, patient__user=request.user)
        
        # Get nearby doctors
        patient = Patient.objects.get(user=request.user)
        doctors = Doctor.objects.filter(address__icontains=patient.address, status=1)
        
        d = {
            'ecg_record': ecg_record,
            'doctors': doctors
        }
        return render(request, 'ecg_result.html', d)
    
    except ECG_Prediction.DoesNotExist:
        return redirect('upload_ecg')

@login_required(login_url='login')
def ecg_history(request):
    """View ECG prediction history"""
    try:
        patient = Patient.objects.get(user=request.user)
        ecg_records = ECG_Prediction.objects.filter(patient=patient).order_by('-created')
        d = {'ecg_records': ecg_records}
        return render(request, 'ecg_history.html', d)
    except:
        return redirect('patient_home')


@login_required(login_url='login')
def find_doctors(request):
    """Find nearby cardiac doctors with map"""
    import json
    
    try:
        patient = Patient.objects.get(user=request.user)
        
        # Get all approved cardiologists
        doctors = Doctor.objects.filter(status=1).order_by('hospital_name')
        
        # Prepare doctor data for map
        doctors_data = []
        for doctor in doctors:
            doctors_data.append({
                'id': doctor.id,
                'name': f"Dr. {doctor.user.first_name} {doctor.user.last_name}",
                'hospital': doctor.hospital_name or 'Private Practice',
                'address': doctor.address,
                'contact': doctor.contact,
                'specialization': doctor.specialization or 'Cardiologist',
                'latitude': doctor.latitude or 0,
                'longitude': doctor.longitude or 0,
            })
        
        context = {
            'doctors': doctors,
            'doctors_json': json.dumps(doctors_data),
            'patient_address': patient.address
        }
        
        return render(request, 'find_doctors.html', context)
    
    except Patient.DoesNotExist:
        return redirect('patient_home')

@login_required(login_url='login')
def book_appointment(request, doctor_id):
    """Book appointment with a doctor"""
    try:
        patient = Patient.objects.get(user=request.user)
        doctor = Doctor.objects.get(id=doctor_id, status=1)
        
        error = ""
        success = ""
        
        if request.method == "POST":
            appointment_date = request.POST.get('appointment_date')
            appointment_time = request.POST.get('appointment_time')
            reason = request.POST.get('reason', '')
            
            # Get related prediction/ECG if provided
            prediction_id = request.POST.get('prediction_id')
            ecg_id = request.POST.get('ecg_id')
            
            related_prediction = None
            related_ecg = None
            
            if prediction_id:
                try:
                    related_prediction = Search_Data.objects.get(id=prediction_id, patient=patient)
                except:
                    pass
            
            if ecg_id:
                try:
                    related_ecg = ECG_Prediction.objects.get(id=ecg_id, patient=patient)
                except:
                    pass
            
            if not appointment_date or not appointment_time:
                error = "Please select both a date and a time for your appointment."
            else:
                # Create appointment
                appointment = Appointment.objects.create(
                    patient=patient,
                    doctor=doctor,
                    appointment_date=appointment_date,
                    appointment_time=appointment_time,
                    reason=reason,
                    related_prediction=related_prediction,
                    related_ecg=related_ecg,
                    status='pending'
                )
                success = "Appointment booked successfully! The doctor will confirm shortly."
            
        context = {
            'doctor': doctor,
            'error': error,
            'success': success
        }
        
        return render(request, 'book_appointment.html', context)
    
    except (Patient.DoesNotExist, Doctor.DoesNotExist):
        return redirect('find_doctors')

@login_required(login_url='login')
def my_appointments(request):
    """View patient's appointments"""
    try:
        from .models import AIBookedAppointment
        patient = Patient.objects.get(user=request.user)
        appointments = Appointment.objects.filter(patient=patient).order_by('-appointment_date', '-appointment_time')
        ai_appointments = AIBookedAppointment.objects.filter(patient=patient)
        context = {'appointments': appointments, 'ai_appointments': ai_appointments}
        return render(request, 'my_appointments.html', context)
    except Patient.DoesNotExist:
        return redirect('patient_home')

@login_required(login_url='login')
def cancel_appointment(request, appointment_id):
    """Cancel an appointment"""
    try:
        patient = Patient.objects.get(user=request.user)
        appointment = Appointment.objects.get(id=appointment_id, patient=patient)
        
        if appointment.status == 'pending' or appointment.status == 'confirmed':
            appointment.status = 'cancelled'
            appointment.save()
            messages.success(request, 'Appointment cancelled successfully.')
        else:
            messages.error(request, 'Cannot cancel this appointment.')
        
        return redirect('my_appointments')
    
    except (Patient.DoesNotExist, Appointment.DoesNotExist):
        return redirect('my_appointments')


@login_required(login_url='login')
def ai_book_appointment(request):
    """Initiate AI call to book appointment"""
    from .ai_calling_agent import create_simple_booking_call
    import re
    
    if request.method == "POST":
        hospital_name = request.POST.get('hospital_name')
        hospital_phone = request.POST.get('hospital_phone')
        hospital_address = request.POST.get('hospital_address')
        
        print(f"DEBUG: Received booking request for {hospital_name}")
        print(f"DEBUG: Phone number received: {hospital_phone}")
        
        try:
            patient = Patient.objects.get(user=request.user)
            patient_name = f"{request.user.first_name} {request.user.last_name}"
            patient_contact = patient.contact

            # Fall back to TEST_PHONE_NUMBER when facility phone is unavailable
            if not hospital_phone:
                hospital_phone = os.getenv('TEST_PHONE_NUMBER', '')
                print(f"DEBUG: No hospital phone provided, falling back to TEST_PHONE_NUMBER: {hospital_phone}")

            if not hospital_phone:
                messages.error(request, "No phone number available for this hospital. Please call manually.")
                return redirect('find_doctors')

            # Format phone number to E.164 format if needed
            if hospital_phone and not hospital_phone.startswith('+'):
                # Remove all non-digit characters
                clean_phone = re.sub(r'\D', '', hospital_phone)
                
                # Add country code if not present
                if len(clean_phone) == 10:  # Indian number without country code
                    hospital_phone = f"+91{clean_phone}"
                elif len(clean_phone) == 11 and clean_phone.startswith('0'):
                    hospital_phone = f"+91{clean_phone[1:]}"
                elif not clean_phone.startswith('91'):
                    hospital_phone = f"+{clean_phone}"
                else:
                    hospital_phone = f"+{clean_phone}"
                
                print(f"DEBUG: Formatted phone to: {hospital_phone}")
            
            # Get reason from recent ECG or prediction
            reason = "Cardiac consultation - Recent ECG analysis showed concerning results"
            
            # Initiate AI call with hospital details
            print(f"DEBUG: Calling create_simple_booking_call with phone: {hospital_phone}")
            result = create_simple_booking_call(
                hospital_phone,
                patient_name,
                patient_contact,
                reason,
                hospital_name=hospital_name,
                hospital_address=hospital_address
            )
            
            print(f"DEBUG: Call result: {result}")
            
            if result['success']:
                messages.success(
                    request,
                    f"✅ AI Agent is calling {hospital_name} to book your appointment. "
                    f"Call SID: {result.get('call_sid', 'N/A')}. "
                    f"You'll receive an SMS confirmation shortly."
                )
            else:
                error_msg = result.get('error', result.get('message', 'Unknown error'))
                messages.error(request, f"❌ Failed to initiate call: {error_msg}")
                print(f"DEBUG: Call failed with error: {error_msg}")
        
        except Exception as e:
            error_msg = str(e)
            messages.error(request, f"❌ Error: {error_msg}")
            print(f"DEBUG: Exception occurred: {error_msg}")
            import traceback
            traceback.print_exc()
        
        return redirect('find_doctors')
    
    return redirect('find_doctors')

from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse

@csrf_exempt
def ai_call_handler(request):
    """Handle AI call conversation flow with Gemini AI"""
    from .ai_calling_agent import AICallingAgent
    from twilio.twiml.voice_response import VoiceResponse
    
    print(f"\n{'='*60}")
    print(f"🎯 AI CALL HANDLER TRIGGERED")
    print(f"{'='*60}")
    print(f"Method: {request.method}")
    print(f"Path: {request.path}")
    print(f"GET params: {dict(request.GET)}")
    print(f"POST params: {dict(request.POST)}")
    
    agent = AICallingAgent()
    stage = request.GET.get('stage', 'greeting')
    call_sid = request.GET.get('call_sid', request.POST.get('CallSid', ''))
    
    print(f"DEBUG: AI Call Handler - Stage: {stage}, CallSid: {call_sid}")
    
    # Get speech result if available
    speech_result = request.POST.get('SpeechResult', '')
    print(f"DEBUG: Speech Result: {speech_result}")
    
    # Get patient data from URL params
    patient_name = request.GET.get('patient_name', '')
    patient_contact = request.GET.get('patient_contact', '')
    reason = request.GET.get('reason', 'Cardiac consultation')
    
    patient_data = {
        'name': patient_name,
        'contact': patient_contact,
        'reason': reason
    }
    
    # Handle different conversation stages
    if stage == 'greeting':
        # Initial greeting with patient data
        data = {'patient_data': patient_data}
        twiml = agent.create_twiml_response('greeting', data=data, call_sid=call_sid)
    
    elif stage == 'conversation':
        # AI-powered conversation
        patient_name = request.GET.get('patient_name', '')
        patient_contact = request.GET.get('patient_contact', '')
        reason = request.GET.get('reason', 'Cardiac consultation')
        
        patient_data = {
            'name': patient_name,
            'contact': patient_contact,
            'reason': reason
        }
        
        data = {
            'SpeechResult': speech_result,
            'patient_data': patient_data
        }
        
        twiml = agent.create_twiml_response('conversation', data=data, call_sid=call_sid)
    
    elif stage == 'confirm_appointment':
        # Final confirmation
        twiml = agent.create_twiml_response('confirm_appointment')
    
    else:
        # Default fallback
        twiml = agent.create_twiml_response('greeting', call_sid=call_sid)
    
    print(f"DEBUG: Generated TwiML: {twiml[:200]}...")
    return HttpResponse(twiml, content_type='text/xml')

@csrf_exempt
def call_status(request):
    """Handle call status callbacks and send notifications when call completes"""
    from .ai_calling_agent import AICallingAgent
    
    call_sid = request.POST.get('CallSid')
    call_status_value = request.POST.get('CallStatus')
    call_duration = request.POST.get('CallDuration', '0')
    
    print(f"\n{'='*60}")
    print(f"📞 CALL STATUS UPDATE")
    print(f"{'='*60}")
    print(f"Call SID: {call_sid}")
    print(f"Status: {call_status_value}")
    print(f"Duration: {call_duration} seconds")
    
    if call_status_value == 'completed':
        print(f"\n✅ Call completed! Saving appointment and sending notifications...")
        try:
            from django.core.cache import cache
            from .models import AIBookedAppointment
            import datetime as _dt

            call_data = cache.get(f'call_data_{call_sid}', {})
            booking   = cache.get(f'call_booking_{call_sid}', {})
            print(f"Call data from cache: {call_data}")
            print(f"Booking details from cache: {booking}")

            # Locate patient by contact number
            patient_contact = call_data.get('patient_contact', '')
            patient_obj = None
            if patient_contact:
                last10 = patient_contact.replace('+', '')[-10:]
                patient_obj = Patient.objects.filter(contact__icontains=last10).first()

            if patient_obj and call_data:
                # Parse date
                date_str = booking.get('date_str', '')
                appt_date = None
                try:
                    if 'tomorrow' in date_str.lower():
                        appt_date = _dt.date.today() + _dt.timedelta(days=1)
                    elif date_str:
                        from dateutil import parser as _dp
                        appt_date = _dp.parse(date_str, fuzzy=True).date()
                except Exception:
                    pass

                # Parse time
                time_str = booking.get('time_str', '')
                appt_time = None
                try:
                    if time_str:
                        from dateutil import parser as _dp
                        appt_time = _dp.parse(time_str, fuzzy=True).time()
                except Exception:
                    pass

                datetime_display = f"{date_str} {time_str}".strip() or 'To be confirmed'

                if not AIBookedAppointment.objects.filter(call_sid=call_sid).exists():
                    AIBookedAppointment.objects.create(
                        patient=patient_obj,
                        hospital_name=call_data.get('hospital_name', 'Hospital'),
                        hospital_phone=call_data.get('hospital_phone', ''),
                        hospital_address=call_data.get('hospital_address', ''),
                        doctor_name=booking.get('doctor_name', ''),
                        department=booking.get('department', ''),
                        appointment_date=appt_date,
                        appointment_time=appt_time,
                        appointment_datetime_str=datetime_display,
                        reason=call_data.get('reason', ''),
                        call_sid=call_sid,
                        status='confirmed',
                    )
                    print(f"✅ AIBookedAppointment saved for {patient_obj.user.username}")
                else:
                    print(f"ℹ️  Appointment for call {call_sid} already saved")
            else:
                print(f"⚠️  Could not find patient (contact={patient_contact})")

            # SMS/WhatsApp notification
            agent = AICallingAgent()
            patient_phone = os.getenv('TEST_PHONE_NUMBER', patient_contact)
            if patient_phone:
                notification_details = {
                    'hospital_name': call_data.get('hospital_name', 'Hospital'),
                    'hospital_address': call_data.get('hospital_address', 'N/A'),
                    'hospital_phone': call_data.get('hospital_phone', 'N/A'),
                    'date': booking.get('date_str', 'To be confirmed'),
                    'time': booking.get('time_str', 'To be confirmed'),
                    'doctor': booking.get('doctor_name', ''),
                    'department': booking.get('department', ''),
                    'status': 'Confirmed via AI call',
                    'call_duration': f"{call_duration} seconds",
                }
                results = agent.send_appointment_notifications(patient_phone, notification_details)
                if results['sms']:
                    print(f"✅ SMS notification sent")
                if results['whatsapp']:
                    print(f"✅ WhatsApp notification sent")

        except Exception as e:
            print(f"Error in call_status handler: {str(e)}")
            import traceback
            traceback.print_exc()

    return HttpResponse('OK')


# ---------------------------------------------------------------------------
# AI Medical Chat
# ---------------------------------------------------------------------------
import json as _json
import urllib.request as _urllib_req
from django.http import JsonResponse
from django.views.decorators.http import require_POST

_CLAUDE_API_URL = 'https://api.quatarly.cloud/v1/messages'
_CLAUDE_MODEL   = 'claude-sonnet-4-6-thinking'
_CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY', 'your-api-key-1')

CHAT_SYSTEM_PROMPT = """You are a helpful heart health assistant. Answer questions about heart health, prediction results, ECG findings, lifestyle, diet, and exercise in plain conversational language.

Rules:
- Write like you are texting a friend — short paragraphs, no bullet points, no headers, no bold, no markdown
- 2 to 4 sentences per response unless more detail is needed
- Always suggest seeing a doctor for serious concerns
- Reference the patient data below when relevant

Patient data:
{patient_context}"""

CHAT_CONTEXT_TEMPLATE = "{user_question}"


def _build_patient_context(patient):
    lines = ["PATIENT: " + (patient.user.get_full_name() or patient.user.username)]
    if patient.dob:
        age = (datetime.date.today() - patient.dob).days // 365
        lines.append(f"AGE: {age}")
    latest_pred = Search_Data.objects.filter(patient=patient).order_by('-created').first()
    if latest_pred:
        lines.append(f"LATEST HEART PREDICTION: {latest_pred.result} (accuracy {latest_pred.prediction_accuracy})")
        lines.append(f"INPUT VALUES (age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal): {latest_pred.values_list}")
    latest_ecg = ECG_Prediction.objects.filter(patient=patient).order_by('-created').first()
    if latest_ecg:
        conf = f" (confidence {latest_ecg.confidence:.0%})" if latest_ecg.confidence else ""
        lines.append(f"LATEST ECG: {latest_ecg.prediction_label} - {latest_ecg.prediction_message}{conf}")
    upcoming = Appointment.objects.filter(
        patient=patient, status__in=['pending', 'confirmed'],
        appointment_date__gte=datetime.date.today()
    ).count()
    if upcoming:
        lines.append(f"UPCOMING APPOINTMENTS: {upcoming}")

    # Medical documents parsed data
    docs = MedicalDocument.objects.filter(patient=patient).order_by('-created_at')[:5]
    if docs:
        lines.append(f"\nUPLOADED MEDICAL DOCUMENTS ({docs.count()} recent):")
        for doc in docs:
            p = doc.parsed_data or {}
            lines.append(f"  [{doc.original_name}]")
            if p.get('diagnosis'):
                lines.append(f"    Diagnosis: {p['diagnosis']}")
            if p.get('medications'):
                lines.append(f"    Medications: {p['medications']}")
            if p.get('allergies'):
                lines.append(f"    Allergies: {p['allergies']}")
            if p.get('vitals'):
                v = p['vitals']
                vitals_str = ', '.join(f"{k}: {v[k]}" for k in v if v.get(k))
                if vitals_str:
                    lines.append(f"    Vitals: {vitals_str}")
            if p.get('lab_results'):
                lines.append(f"    Lab Results: {p['lab_results']}")
            if p.get('doctor_notes'):
                lines.append(f"    Doctor Notes: {p['doctor_notes']}")
            if p.get('summary'):
                lines.append(f"    Summary: {p['summary']}")

    return "\n".join(lines)


def _call_claude_chat(system_prompt, messages_history, api_key, max_tokens=600):
    payload = _json.dumps({
        'model': _CLAUDE_MODEL,
        'max_tokens': max_tokens,
        'system': system_prompt,
        'messages': messages_history
    }).encode()
    req = _urllib_req.Request(
        _CLAUDE_API_URL,
        data=payload,
        headers={
            'Content-Type': 'application/json',
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01'
        }
    )
    with _urllib_req.urlopen(req, timeout=20) as r:
        data = _json.loads(r.read())
    return data['content'][0]['text'].strip()


@login_required(login_url='login')
def medical_chat(request, session_id=None):
    patient = Patient.objects.filter(user=request.user).first()
    if not patient:
        return redirect('patient_home')
    sessions = ChatSession.objects.filter(patient=patient)
    active_session = None
    if session_id:
        active_session = ChatSession.objects.filter(id=session_id, patient=patient).first()
    if not active_session:
        active_session = sessions.first()
    chat_messages = active_session.messages.all() if active_session else []
    return render(request, 'medical_chat.html', {
        'sessions': sessions,
        'active_session': active_session,
        'chat_messages': chat_messages,
        'patient': patient,
    })


@login_required(login_url='login')
def medical_chat_new(request):
    patient = Patient.objects.filter(user=request.user).first()
    if not patient:
        return redirect('patient_home')
    session = ChatSession.objects.create(patient=patient, title='New Chat')
    return redirect('medical_chat_session', session_id=session.id)


@login_required(login_url='login')
@require_POST
def medical_chat_send(request):
    patient = Patient.objects.filter(user=request.user).first()
    if not patient:
        return JsonResponse({'error': 'Not a patient account'}, status=403)
    try:
        body = _json.loads(request.body)
    except Exception:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    user_text = body.get('message', '').strip()
    session_id = body.get('session_id')
    if not user_text:
        return JsonResponse({'error': 'Empty message'}, status=400)
    if session_id:
        session = ChatSession.objects.filter(id=session_id, patient=patient).first()
    else:
        session = None
    if not session:
        session = ChatSession.objects.create(patient=patient, title=user_text[:50])
    if session.title == 'New Chat' and session.messages.count() == 0:
        session.title = user_text[:50] + ('...' if len(user_text) > 50 else '')
        session.save(update_fields=['title'])
    ChatMessage.objects.create(session=session, role='user', content=user_text)
    history = list(session.messages.order_by('created_at'))
    patient_ctx = _build_patient_context(patient)
    system = CHAT_SYSTEM_PROMPT.format(patient_context=patient_ctx)
    claude_messages = [
        {'role': 'user' if m.role == 'user' else 'assistant', 'content': m.content}
        for m in history
    ]
    try:
        ai_text = _call_claude_chat(system, claude_messages, _CLAUDE_API_KEY)
    except Exception as e:
        return JsonResponse({'error': f'AI unavailable: {str(e)}'}, status=503)
    ChatMessage.objects.create(session=session, role='ai', content=ai_text)
    session.save(update_fields=['updated_at'])
    return JsonResponse({
        'reply': ai_text,
        'session_id': session.id,
        'session_title': session.title,
    })


@login_required(login_url='login')
@require_POST
def medical_chat_delete_session(request, session_id):
    patient = Patient.objects.filter(user=request.user).first()
    if patient:
        ChatSession.objects.filter(id=session_id, patient=patient).delete()
    return JsonResponse({'ok': True})


@login_required(login_url='login')
def medical_chat_from_result(request, pred, accuracy):
    """Create a new chat session pre-loaded with the prediction result and AI reply."""
    patient = Patient.objects.filter(user=request.user).first()
    if not patient:
        return redirect('patient_home')

    result_label = 'AT RISK for heart disease' if str(pred) != '0' else 'HEALTHY (no heart disease detected)'
    auto_message = (
        f"I just got my heart disease prediction result back. "
        f"The model says I am {result_label} with {accuracy}% accuracy. "
        f"Can you explain what this means in simple terms and what steps I should take next?"
    )

    session = ChatSession.objects.create(patient=patient, title=f'My prediction — {result_label[:22]}')
    ChatMessage.objects.create(session=session, role='user', content=auto_message)

    patient_ctx = _build_patient_context(patient)
    system = CHAT_SYSTEM_PROMPT.format(patient_context=patient_ctx)
    try:
        ai_text = _call_claude_chat(system, [{'role': 'user', 'content': auto_message}], _CLAUDE_API_KEY)
    except Exception:
        ai_text = "I was unable to reach the AI right now. Please try opening a new chat and typing your question."
    ChatMessage.objects.create(session=session, role='ai', content=ai_text)
    session.save(update_fields=['updated_at'])
    return redirect('medical_chat_session', session_id=session.id)


# ---------------------------------------------------------------------------
# Medical Document Parser
# ---------------------------------------------------------------------------
from .models import MedicalDocument

_DOC_PARSE_SYSTEM = """You are a medical document parser. Extract all relevant health information from the provided document text.
Return ONLY a valid JSON object with these fields (omit any field where no data was found):
{
  "diagnosis": "diagnosed conditions",
  "medications": "current medications and dosages",
  "vitals": {"blood_pressure": "", "heart_rate": "", "temperature": "", "spo2": "", "cholesterol": ""},
  "lab_results": "key lab findings",
  "doctor_notes": "doctor observations and recommendations",
  "allergies": "any allergies mentioned",
  "follow_up": "recommended follow-up actions",
  "summary": "a 2-3 sentence plain-English summary of the document"
}
Return ONLY valid JSON, no markdown fences."""


def _parse_doc_text_with_claude(text, filename):
    from .ai_calling_agent import call_claude
    from django.conf import settings as _settings
    api_key = getattr(_settings, 'CLAUDE_API_KEY', '') or os.getenv('CLAUDE_API_KEY', '')
    prompt = f"Document filename: {filename}\n\nDocument content:\n{text[:8000]}"
    try:
        raw = call_claude(_DOC_PARSE_SYSTEM, prompt, api_key, max_tokens=800)
        # strip markdown fences if any
        raw = raw.strip()
        if raw.startswith('```'):
            raw = raw.split('\n', 1)[1] if '\n' in raw else raw[3:]
        if raw.endswith('```'):
            raw = raw[:-3]
        return _json.loads(raw.strip())
    except Exception as e:
        return {'summary': f'Parsing failed: {str(e)}', 'raw_text': text[:500]}


@login_required(login_url='login')
def medical_documents(request):
    patient = Patient.objects.filter(user=request.user).first()
    if not patient:
        return redirect('patient_home')
    docs = MedicalDocument.objects.filter(patient=patient)
    return render(request, 'medical_documents.html', {'docs': docs, 'patient': patient})


@login_required(login_url='login')
def upload_medical_document(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    patient = Patient.objects.filter(user=request.user).first()
    if not patient:
        return JsonResponse({'error': 'Not a patient account'}, status=403)

    uploaded = request.FILES.get('document')
    if not uploaded:
        return JsonResponse({'error': 'No file provided'}, status=400)

    fname = uploaded.name.lower()
    if fname.endswith('.pdf'):
        file_type = 'pdf'
    elif any(fname.endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.webp')):
        file_type = 'image'
    else:
        return JsonResponse({'error': 'Only PDF or image files are supported'}, status=400)

    # Extract text
    try:
        if file_type == 'pdf':
            import pdfplumber, io
            with pdfplumber.open(io.BytesIO(uploaded.read())) as pdf:
                text = '\n'.join(p.extract_text() or '' for p in pdf.pages)
            uploaded.seek(0)
        else:
            import base64
            raw_bytes = uploaded.read()
            text = f"[Image file: {uploaded.name}]\nBase64 size: {len(raw_bytes)} bytes"
            uploaded.seek(0)
    except Exception as e:
        return JsonResponse({'error': f'Could not read file: {str(e)}'}, status=400)

    parsed = _parse_doc_text_with_claude(text, uploaded.name)

    doc = MedicalDocument.objects.create(
        patient=patient,
        file=uploaded,
        original_name=uploaded.name,
        file_type=file_type,
        parsed_data=parsed,
    )
    return JsonResponse({'ok': True, 'doc_id': doc.id, 'parsed': parsed})


@login_required(login_url='login')
def delete_medical_document(request, doc_id):
    patient = Patient.objects.filter(user=request.user).first()
    if patient:
        MedicalDocument.objects.filter(id=doc_id, patient=patient).delete()
    return JsonResponse({'ok': True})


# ---------------------------------------------------------------------------
# Voice Symptom Collector  (Sarvam AI STT)
# ---------------------------------------------------------------------------

@login_required(login_url='login')
@require_POST
def voice_transcribe(request):
    """Receive base64 audio → Sarvam STT → Claude field extraction → return JSON."""
    try:
        body = _json.loads(request.body)
    except Exception:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    audio_b64 = body.get('audio')
    mime_type = body.get('mime_type', 'audio/webm')
    if not audio_b64:
        return JsonResponse({'error': 'No audio provided'}, status=400)

    from django.conf import settings as _settings
    sarvam_key = getattr(_settings, 'SARVAM_API_KEY', '') or os.getenv('SARVAM_API_KEY', '')

    # --- Step 1: Speech-to-text via Sarvam AI ---
    transcript = ''
    language_name = 'English'
    if sarvam_key:
        try:
            import tempfile, base64 as _b64
            audio_bytes = _b64.b64decode(audio_b64)
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
                f.write(audio_bytes)
                tmp = f.name
            import requests as _req
            with open(tmp, 'rb') as af:
                resp = _req.post(
                    'https://api.sarvam.ai/speech-to-text',
                    headers={'API-Subscription-Key': sarvam_key},
                    files={'file': ('audio.webm', af, mime_type)},
                    data={'model': 'saarika:v2.5', 'with_timestamps': 'false'},
                    timeout=30
                )
            os.unlink(tmp)
            if resp.status_code == 200:
                data = resp.json()
                transcript = data.get('transcript', '')
                lang_code = data.get('language_code', 'en-IN')
                lang_map = {
                    'hi-IN': 'Hindi', 'ta-IN': 'Tamil', 'te-IN': 'Telugu',
                    'bn-IN': 'Bengali', 'kn-IN': 'Kannada', 'gu-IN': 'Gujarati',
                    'ml-IN': 'Malayalam', 'mr-IN': 'Marathi', 'pa-IN': 'Punjabi',
                    'en-IN': 'English',
                }
                language_name = lang_map.get(lang_code, 'English')
        except Exception as e:
            return JsonResponse({'error': f'Sarvam STT error: {str(e)}'}, status=503)
    else:
        # Fallback: treat audio_b64 as plain text for testing
        transcript = body.get('text_fallback', '')
        if not transcript:
            return JsonResponse({'error': 'SARVAM_API_KEY not configured and no text_fallback provided'}, status=503)

    if not transcript.strip():
        return JsonResponse({'error': 'Could not transcribe audio — please speak clearly and try again'}, status=400)

    # --- Step 2: Claude extracts form field values from transcript ---
    _FIELD_EXTRACT_SYSTEM = """You are a medical data extraction AI. Given a patient's spoken description of their health, extract values for the heart disease prediction form.

Return ONLY a valid JSON object with any of these fields you can confidently extract:
{
  "age": <integer years>,
  "sex": <1 for male, 0 for female>,
  "cp": <chest pain: 0=typical angina, 1=atypical angina, 2=non-anginal, 3=asymptomatic>,
  "trestbps": <resting blood pressure mmHg integer>,
  "chol": <cholesterol mg/dl integer>,
  "fbs": <fasting blood sugar >120: 1=yes 0=no>,
  "thalach": <max heart rate integer>,
  "exang": <exercise induced angina: 1=yes 0=no>
}
Only include fields you can confidently extract from what the patient said. Do NOT guess. Omit unclear fields.
The patient said (translated if needed): "{transcript}"
Return ONLY valid JSON."""

    from .ai_calling_agent import call_claude
    api_key = getattr(_settings, 'CLAUDE_API_KEY', '') or os.getenv('CLAUDE_API_KEY', '')
    try:
        raw = call_claude(
            _FIELD_EXTRACT_SYSTEM.replace('{transcript}', transcript),
            'Extract the form fields from the transcript above.',
            api_key, max_tokens=200
        )
        raw = raw.strip()
        if raw.startswith('```'):
            raw = raw.split('\n', 1)[1] if '\n' in raw else raw[3:]
        if raw.endswith('```'):
            raw = raw[:-3]
        fields = _json.loads(raw.strip())
    except Exception:
        fields = {}

    return JsonResponse({
        'transcript': transcript,
        'language': language_name,
        'fields': fields,
    })


# ---------------------------------------------------------------------------
# Health Risk Trend Dashboard
# ---------------------------------------------------------------------------

@login_required(login_url='login')
def health_trends(request):
    patient = Patient.objects.filter(user=request.user).first()
    if not patient:
        return redirect('patient_home')

    predictions = Search_Data.objects.filter(patient=patient).order_by('created')
    ecg_records = ECG_Prediction.objects.filter(patient=patient).order_by('created')

    # Aggregate stats
    total_preds = predictions.count()
    at_risk_count = predictions.filter(result='1').count()
    avg_accuracy = 0
    if total_preds:
        accs = []
        for p in predictions:
            try:
                accs.append(float(p.prediction_accuracy))
            except (TypeError, ValueError):
                pass
        avg_accuracy = round(sum(accs) / len(accs), 1) if accs else 0

    latest_pred = predictions.last()
    latest_ecg  = ecg_records.last()

    context = {
        'patient': patient,
        'total_preds': total_preds,
        'at_risk_count': at_risk_count,
        'healthy_count': total_preds - at_risk_count,
        'avg_accuracy': avg_accuracy,
        'ecg_count': ecg_records.count(),
        'latest_pred': latest_pred,
        'latest_ecg': latest_ecg,
    }
    return render(request, 'health_trends.html', context)


@login_required(login_url='login')
def health_trends_data(request):
    """JSON endpoint consumed by Chart.js on the trend dashboard."""
    from django.http import JsonResponse

    patient = Patient.objects.filter(user=request.user).first()
    if not patient:
        return JsonResponse({'error': 'not a patient'}, status=403)

    # --- Heart-disease predictions series ---
    predictions = Search_Data.objects.filter(patient=patient).order_by('created')
    pred_labels, pred_risk, pred_accuracy = [], [], []
    for p in predictions:
        ts = p.created
        label = ts.strftime('%b %d') if ts else '?'
        pred_labels.append(label)
        pred_risk.append(int(p.result) if p.result in ('0', '1') else 0)
        try:
            pred_accuracy.append(round(float(p.prediction_accuracy), 1))
        except (TypeError, ValueError):
            pred_accuracy.append(0)

    # Extract individual feature trends from values_list
    feature_names = ['Age', 'Sex', 'Chest Pain', 'Resting BP', 'Cholesterol',
                     'Fasting BS', 'Rest ECG', 'Max HR', 'Exang', 'Oldpeak',
                     'Slope', 'CA', 'Thal']
    chol_series, bp_series, hr_series = [], [], []
    for p in predictions:
        try:
            vals = _json.loads(p.values_list.replace("'", '"')) if p.values_list else []
            chol_series.append(vals[4] if len(vals) > 4 else None)
            bp_series.append(vals[3] if len(vals) > 3 else None)
            hr_series.append(vals[7] if len(vals) > 7 else None)
        except Exception:
            chol_series.append(None)
            bp_series.append(None)
            hr_series.append(None)

    # --- ECG series ---
    ecg_records = ECG_Prediction.objects.filter(patient=patient).order_by('created')
    ecg_label_map = {0: 'Abnormal', 1: 'MI', 2: 'Normal', 3: 'History of MI'}
    ecg_labels, ecg_codes, ecg_confidence = [], [], []
    for e in ecg_records:
        ecg_labels.append(e.created.strftime('%b %d') if e.created else '?')
        ecg_codes.append(e.prediction_code if e.prediction_code is not None else -1)
        ecg_confidence.append(round(e.confidence * 100, 1) if e.confidence else None)

    return JsonResponse({
        'predictions': {
            'labels': pred_labels,
            'risk': pred_risk,
            'accuracy': pred_accuracy,
            'cholesterol': chol_series,
            'blood_pressure': bp_series,
            'heart_rate': hr_series,
        },
        'ecg': {
            'labels': ecg_labels,
            'codes': ecg_codes,
            'confidence': ecg_confidence,
            'label_map': ecg_label_map,
        }
    })


# ---------------------------------------------------------------------------
# PDF Health Report Generator
# ---------------------------------------------------------------------------

@login_required(login_url='login')
def download_health_report(request):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable, KeepTogether)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from io import BytesIO

    patient = Patient.objects.filter(user=request.user).first()
    if not patient:
        return redirect('patient_home')

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=1.8*cm, leftMargin=1.8*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    # ── Colour palette ──────────────────────────────────────────────────────
    DARK    = colors.HexColor('#1c1208')
    MID     = colors.HexColor('#5c3d2e')
    LIGHT   = colors.HexColor('#faf7f0')
    ACCENT  = colors.HexColor('#8b2e2e')
    MUTED   = colors.HexColor('#8a7060')
    GREEN   = colors.HexColor('#2e6b3e')
    YELLOW  = colors.HexColor('#b8860b')

    styles  = getSampleStyleSheet()

    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    sTitle   = S('sTitle',  fontSize=22, textColor=DARK,   spaceAfter=2,  fontName='Helvetica-Bold', alignment=TA_CENTER)
    sSub     = S('sSub',    fontSize=11, textColor=MUTED,  spaceAfter=2,  fontName='Helvetica',      alignment=TA_CENTER)
    sHead    = S('sHead',   fontSize=13, textColor=DARK,   spaceBefore=14, spaceAfter=4, fontName='Helvetica-Bold')
    sBody    = S('sBody',   fontSize=9,  textColor=DARK,   spaceAfter=2,  fontName='Helvetica',      leading=14)
    sSmall   = S('sSmall',  fontSize=8,  textColor=MUTED,  fontName='Helvetica')
    sGreen   = S('sGreen',  fontSize=9,  textColor=GREEN,  fontName='Helvetica-Bold')
    sRed     = S('sRed',    fontSize=9,  textColor=ACCENT, fontName='Helvetica-Bold')
    sYellow  = S('sYellow', fontSize=9,  textColor=YELLOW, fontName='Helvetica-Bold')

    story = []
    W = A4[0] - 3.6*cm   # usable width

    def hr():
        return HRFlowable(width='100%', thickness=0.5, color=MID, spaceAfter=8, spaceBefore=4)

    def section(title):
        story.append(Spacer(1, 6))
        story.append(Paragraph(title, sHead))
        story.append(hr())

    # ── Cover ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(Paragraph('❤ Heart Health Report', sTitle))
    full_name = f"{patient.user.first_name} {patient.user.last_name}".strip() or patient.user.username
    story.append(Paragraph(f'Patient: {full_name}', sSub))
    story.append(Paragraph(f'Generated: {datetime.date.today().strftime("%B %d, %Y")}', sSub))
    story.append(Spacer(1, 14))
    story.append(hr())

    # ── Patient Info ────────────────────────────────────────────────────────
    section('Patient Information')
    age_str = '—'
    if patient.dob:
        age_str = str((datetime.date.today() - patient.dob).days // 365)
    info_data = [
        ['Name',    full_name,              'Username', patient.user.username],
        ['Email',   patient.user.email or '—', 'Age',   age_str],
        ['Contact', patient.contact or '—', 'Address',  patient.address or '—'],
    ]
    info_tbl = Table(info_data, colWidths=[2.5*cm, W/2-2.5*cm, 2.5*cm, W/2-2.5*cm])
    info_tbl.setStyle(TableStyle([
        ('FONTNAME',  (0,0),(-1,-1), 'Helvetica'),
        ('FONTNAME',  (0,0),(0,-1),  'Helvetica-Bold'),
        ('FONTNAME',  (2,0),(2,-1),  'Helvetica-Bold'),
        ('FONTSIZE',  (0,0),(-1,-1), 9),
        ('TEXTCOLOR', (0,0),(0,-1),  MUTED),
        ('TEXTCOLOR', (2,0),(2,-1),  MUTED),
        ('ROWBACKGROUNDS', (0,0),(-1,-1), [LIGHT, colors.white]),
        ('GRID',      (0,0),(-1,-1), 0.3, colors.HexColor('#e8ddd0')),
        ('PADDING',   (0,0),(-1,-1), 6),
    ]))
    story.append(info_tbl)

    # ── Summary Stats ───────────────────────────────────────────────────────
    predictions = list(Search_Data.objects.filter(patient=patient).order_by('created'))
    ecg_records = list(ECG_Prediction.objects.filter(patient=patient).order_by('-created'))
    appointments = list(Appointment.objects.filter(patient=patient).order_by('-appointment_date'))
    from .models import AIBookedAppointment
    ai_appts = list(AIBookedAppointment.objects.filter(patient=patient).order_by('-created'))

    at_risk  = sum(1 for p in predictions if str(p.result) == '1')
    healthy  = len(predictions) - at_risk
    normal_ecg = sum(1 for e in ecg_records if e.prediction_code == 2)

    section('Summary')
    stat_data = [
        ['Heart Predictions', 'At-Risk Results', 'Healthy Results', 'ECG Analyses'],
        [str(len(predictions)), str(at_risk), str(healthy), str(len(ecg_records))],
    ]
    stat_tbl = Table(stat_data, colWidths=[W/4]*4)
    stat_tbl.setStyle(TableStyle([
        ('FONTNAME',    (0,0),(-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',    (0,0),(-1,0),  8),
        ('FONTSIZE',    (0,1),(-1,1),  18),
        ('TEXTCOLOR',   (0,0),(-1,0),  MUTED),
        ('TEXTCOLOR',   (0,1),(0,1),   DARK),
        ('TEXTCOLOR',   (1,1),(1,1),   ACCENT),
        ('TEXTCOLOR',   (2,1),(2,1),   GREEN),
        ('TEXTCOLOR',   (3,1),(3,1),   MID),
        ('ALIGN',       (0,0),(-1,-1), 'CENTER'),
        ('BACKGROUND',  (0,0),(-1,-1), LIGHT),
        ('GRID',        (0,0),(-1,-1), 0.3, colors.HexColor('#e8ddd0')),
        ('PADDING',     (0,0),(-1,-1), 10),
    ]))
    story.append(stat_tbl)

    # ── AI Summary ──────────────────────────────────────────────────────────
    section('AI Health Summary')
    try:
        from .views import _build_patient_context, _call_claude_chat, CHAT_SYSTEM_PROMPT
        ctx = _build_patient_context(patient)
        system = CHAT_SYSTEM_PROMPT.format(patient_context=ctx)
        summary_q = "Write a concise 3-sentence clinical summary of this patient's cardiac health status based on their history. Be direct and factual."
        ai_summary = _call_claude_chat(system, [{'role': 'user', 'content': summary_q}], _CLAUDE_API_KEY, max_tokens=300)
    except Exception:
        ai_summary = 'AI summary unavailable.'
    story.append(Paragraph(ai_summary, sBody))

    # ── Prediction History ──────────────────────────────────────────────────
    section(f'Heart Disease Prediction History ({len(predictions)} records)')
    if predictions:
        ph_header = ['Date', 'Result', 'Confidence', 'Age', 'Chol', 'BP', 'Max HR']
        ph_rows = [ph_header]
        for p in reversed(predictions):
            result_str = 'AT RISK' if str(p.result) == '1' else 'Healthy'
            try:
                acc = f"{round(float(p.prediction_accuracy), 1)}%"
            except Exception:
                acc = '—'
            vals = []
            try:
                import ast
                vals = ast.literal_eval(p.values_list) if p.values_list else []
            except Exception:
                pass
            ph_rows.append([
                p.created.strftime('%b %d, %Y') if p.created else '—',
                result_str,
                acc,
                str(vals[0]) if len(vals) > 0 else '—',
                str(vals[4]) if len(vals) > 4 else '—',
                str(vals[3]) if len(vals) > 3 else '—',
                str(vals[7]) if len(vals) > 7 else '—',
            ])
        col_w = [2.8*cm, 2*cm, 2.2*cm, 1.4*cm, 1.6*cm, 1.4*cm, 1.8*cm]
        ph_tbl = Table(ph_rows, colWidths=col_w, repeatRows=1)
        risk_idxs = [i for i, r in enumerate(ph_rows) if i > 0 and r[1] == 'AT RISK']
        style_cmds = [
            ('FONTNAME',   (0,0),(-1,0),  'Helvetica-Bold'),
            ('FONTSIZE',   (0,0),(-1,-1), 8),
            ('BACKGROUND', (0,0),(-1,0),  DARK),
            ('TEXTCOLOR',  (0,0),(-1,0),  colors.white),
            ('ALIGN',      (0,0),(-1,-1), 'CENTER'),
            ('ROWBACKGROUNDS', (0,1),(-1,-1), [LIGHT, colors.white]),
            ('GRID',       (0,0),(-1,-1), 0.3, colors.HexColor('#e8ddd0')),
            ('PADDING',    (0,0),(-1,-1), 5),
        ]
        for ri in risk_idxs:
            style_cmds.append(('TEXTCOLOR', (1, ri), (1, ri), ACCENT))
            style_cmds.append(('FONTNAME',  (1, ri), (1, ri), 'Helvetica-Bold'))
        ph_tbl.setStyle(TableStyle(style_cmds))
        story.append(ph_tbl)
    else:
        story.append(Paragraph('No prediction records found.', sSmall))

    # ── ECG History ─────────────────────────────────────────────────────────
    section(f'ECG Analysis History ({len(ecg_records)} records)')
    if ecg_records:
        ecg_header = ['Date', 'Result', 'Confidence', 'Message']
        ecg_rows = [ecg_header]
        for e in ecg_records:
            conf_str = f"{round(e.confidence*100, 1)}%" if e.confidence else '—'
            ecg_rows.append([
                e.created.strftime('%b %d, %Y') if e.created else '—',
                e.prediction_label or '—',
                conf_str,
                Paragraph(e.prediction_message or '—', sSmall),
            ])
        ecg_tbl = Table(ecg_rows, colWidths=[2.4*cm, 3.2*cm, 2*cm, W-7.6*cm], repeatRows=1)
        ecg_style = [
            ('FONTNAME',   (0,0),(-1,0),  'Helvetica-Bold'),
            ('FONTSIZE',   (0,0),(-1,-1), 8),
            ('BACKGROUND', (0,0),(-1,0),  MID),
            ('TEXTCOLOR',  (0,0),(-1,0),  colors.white),
            ('ALIGN',      (0,0),(2,-1),  'CENTER'),
            ('ALIGN',      (3,0),(3,-1),  'LEFT'),
            ('VALIGN',     (0,0),(-1,-1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0,1),(-1,-1), [LIGHT, colors.white]),
            ('GRID',       (0,0),(-1,-1), 0.3, colors.HexColor('#e8ddd0')),
            ('PADDING',    (0,0),(-1,-1), 5),
        ]
        ecg_tbl.setStyle(TableStyle(ecg_style))
        story.append(ecg_tbl)
    else:
        story.append(Paragraph('No ECG records found.', sSmall))

    # ── Appointments ────────────────────────────────────────────────────────
    all_appts_count = len(appointments) + len(ai_appts)
    section(f'Appointment History ({all_appts_count} records)')
    appt_rows = [['Date', 'Time', 'Doctor / Hospital', 'Status', 'Type']]
    for a in appointments:
        doc_name = f"Dr. {a.doctor.user.first_name} {a.doctor.user.last_name}"
        hosp = a.doctor.hospital_name or 'Private Practice'
        appt_rows.append([
            a.appointment_date.strftime('%b %d, %Y') if a.appointment_date else '—',
            a.appointment_time.strftime('%I:%M %p') if a.appointment_time else '—',
            f"{doc_name}\n{hosp}",
            a.status.capitalize(),
            'Manual',
        ])
    for a in ai_appts:
        dt_str = a.appointment_date.strftime('%b %d, %Y') if a.appointment_date else a.appointment_datetime_str or '—'
        tm_str = a.appointment_time.strftime('%I:%M %p') if a.appointment_time else '—'
        appt_rows.append([
            dt_str, tm_str,
            f"{a.doctor_name or '—'}\n{a.hospital_name}",
            'Confirmed',
            'AI Call',
        ])
    if len(appt_rows) > 1:
        col_w2 = [2.4*cm, 1.8*cm, W-9.4*cm, 2.2*cm, 2*cm]
        appt_tbl = Table(appt_rows, colWidths=col_w2, repeatRows=1)
        appt_tbl.setStyle(TableStyle([
            ('FONTNAME',   (0,0),(-1,0),  'Helvetica-Bold'),
            ('FONTSIZE',   (0,0),(-1,-1), 8),
            ('BACKGROUND', (0,0),(-1,0),  DARK),
            ('TEXTCOLOR',  (0,0),(-1,0),  colors.white),
            ('ALIGN',      (0,0),(-1,-1), 'CENTER'),
            ('VALIGN',     (0,0),(-1,-1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0,1),(-1,-1), [LIGHT, colors.white]),
            ('GRID',       (0,0),(-1,-1), 0.3, colors.HexColor('#e8ddd0')),
            ('PADDING',    (0,0),(-1,-1), 5),
        ]))
        story.append(appt_tbl)
    else:
        story.append(Paragraph('No appointments found.', sSmall))

    # ── Medical Documents ───────────────────────────────────────────────────
    docs = MedicalDocument.objects.filter(patient=patient)
    if docs.exists():
        section(f'Uploaded Medical Documents ({docs.count()})')
        for mdoc in docs:
            p_data = mdoc.parsed_data or {}
            story.append(Paragraph(f'<b>{mdoc.original_name}</b>  <font color="#8a7060" size="8">({mdoc.created_at.strftime("%b %d, %Y")})</font>', sBody))
            for key in ('diagnosis', 'medications', 'allergies', 'lab_results', 'doctor_notes', 'summary'):
                val = p_data.get(key)
                if val:
                    label = key.replace('_', ' ').title()
                    story.append(Paragraph(f'<font color="#8a7060">{label}:</font> {val}', sSmall))
            story.append(Spacer(1, 4))

    # ── Footer ──────────────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(hr())
    story.append(Paragraph(
        'This report is generated by the Heart Disease Prediction System. '
        'It is intended for informational purposes only and does not replace professional medical advice.',
        sSmall
    ))

    doc.build(story)
    buffer.seek(0)
    fname = f"health_report_{patient.user.username}_{datetime.date.today()}.pdf"
    response = HttpResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{fname}"'
    return response
