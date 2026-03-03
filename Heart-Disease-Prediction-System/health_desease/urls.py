"""health_desease URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from health.views import *
# from .api import router
from .apirep import routerep

urlpatterns = [
    # path('api/sensoviz/', include(router.urls)),
    path('api/v1/', include(routerep.urls)),
    path('admin/', admin.site.urls),
    path('', Home, name="home"),
    path('patient_home', User_Home,name="patient_home"),
    path('doctor_home', Doctor_Home,name="doctor_home"),
    path('admin_home', Admin_Home,name="admin_home"),
    path('gallery', Gallery,name="gallery"),
    path('login', Login_User,name="login"),
    path('login_admin', Login_admin,name="login_admin"),
    path('signup', Signup_User,name="signup"),
    path('logout', Logout,name="logout"),
    path('change_password', Change_Password,name="change_password"),
    # path('prdict_heart_disease', prdict_heart_disease,name="prdict_heart_disease"),
    path('add_heartdetail', add_heartdetail,name="add_heartdetail"),
    path('view_search_pat', view_search_pat,name="view_search_pat"),

    path('view_doctor', View_Doctor,name="view_doctor"),
    path('add_doctor', add_doctor,name="add_doctor"),
    path('change_doctor/<int:pid>/', add_doctor,name="change_doctor"),
    path('view_patient', View_Patient,name="view_patient"),
    path('view_feedback', View_Feedback,name="view_feedback"),
    path('edit_profile', Edit_My_deatail,name="edit_profile"),
    path('profile_doctor', View_My_Detail,name="profile_doctor"),
    path('sent_feedback', sent_feedback,name="sent_feedback"),

    path('delete_searched/<int:pid>', delete_searched, name="delete_searched"),
    path('delete_doctor<int:pid>', delete_doctor, name="delete_doctor"),
    path('assign_status<int:pid>', assign_status, name="assign_status"),
    path('delete_patient<int:pid>', delete_patient, name="delete_patient"),
    path('delete_feedback<int:pid>', delete_feedback, name="delete_feedback"),
    path('predict_desease/<str:pred>/<str:accuracy>/', predict_desease, name="predict_desease"),
    
    # ECG Prediction URLs
    path('upload_ecg', upload_ecg, name="upload_ecg"),
    path('ecg_result/<int:ecg_id>/', ecg_result, name="ecg_result"),
    path('ecg_history', ecg_history, name="ecg_history"),
    
    # Doctor Finder & Appointments
    path('find_doctors', find_doctors, name="find_doctors"),
    path('book_appointment/<int:doctor_id>/', book_appointment, name="book_appointment"),
    path('my_appointments', my_appointments, name="my_appointments"),
    path('cancel_appointment/<int:appointment_id>/', cancel_appointment, name="cancel_appointment"),
    
    # AI Calling Agent
    path('ai_book_appointment', ai_book_appointment, name="ai_book_appointment"),
    path('ai_call_handler/', ai_call_handler, name="ai_call_handler"),
    path('call_status/', call_status, name="call_status"),

    # AI Features
    path('ai_patient_summary/<int:patient_id>/', ai_patient_summary, name="ai_patient_summary"),
    path('medical_chat_from_result/<str:pred>/<str:accuracy>/', medical_chat_from_result, name="medical_chat_from_result"),

    # Medical Document Parser
    path('medical_documents', medical_documents, name="medical_documents"),
    path('upload_medical_document', upload_medical_document, name="upload_medical_document"),
    path('delete_medical_document/<int:doc_id>/', delete_medical_document, name="delete_medical_document"),

    # Voice Symptom Collector
    path('voice_transcribe', voice_transcribe, name="voice_transcribe"),

    # Health Trends Dashboard & PDF Report
    path('health_trends', health_trends, name="health_trends"),
    path('health_trends_data/', health_trends_data, name="health_trends_data"),
    path('download_health_report', download_health_report, name="download_health_report"),

    # AI Medical Chat
    path('medical_chat', medical_chat, name="medical_chat"),
    path('medical_chat/<int:session_id>/', medical_chat, name="medical_chat_session"),
    path('medical_chat_new', medical_chat_new, name="medical_chat_new"),
    path('medical_chat_send', medical_chat_send, name="medical_chat_send"),
    path('medical_chat_delete/<int:session_id>/', medical_chat_delete_session, name="medical_chat_delete_session"),

]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
