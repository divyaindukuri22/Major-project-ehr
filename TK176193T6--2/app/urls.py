from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('about/', views.about, name='about'), 
    path('patientregister/', views.patientregister, name='patientregister'),
    path('patientlogin/', views.patientlogin, name='patientlogin'),
    path('doctorregister/', views.doctorregister, name='doctorregister'),
    path('doctorlogin/', views.doctorlogin, name='doctorlogin'),
    path('adminlogin/', views.adminlogin, name='adminlogin'),

    path('home/', views.home, name='home'),
    path('logout/', views.logout, name='logout'),
    path('chatbot_response/', views.chatbot_response, name='chatbot_response'),
    path('doctors/<str:disease>/', views.doctors, name='doctors'),
    path('contact/', views.contact, name='contact'),
    path('profile/', views.profile, name='profile'),
    path('editprofile/', views.editprofile, name='editprofile'),
    path('sendappointment/<int:id>/', views.sendappointment, name='sendappointment'),
    path('myappointments/', views.myappointments, name='myappointments'),
    path('withdraw/<int:id>/', views.withdraw, name='withdraw'),
    path('viewappointments/', views.viewappointments, name='viewappointments'),
    path('accept/<int:id>/', views.accept, name='accept'),
    path('downloadmri/', views.downloadmri, name='downloadmri'),
    path('viewreports/', views.viewreports, name='viewreports'),
    path('report/<int:id>/', views.report, name='report'),
    path('uploadreport/<int:id>/', views.uploadreport, name='uploadreport'),
    path('myreports/', views.myreports, name='myreports'),
    path('getreport/<int:id>/', views.getreport, name='getreport'),
    path('download/<int:id>/', views.download, name='download'),
    path('feedback/', views.feedback, name='feedback'),
    path('freequote/', views.freequote, name='freequote'),
    path('viewdoctors/', views.viewdoctors, name='viewdoctors'),
    path('authorize/<int:id>/', views.authorize, name='authorize'),
    path('appointments/', views.appointments, name='appointments'),
    
    path('send/<int:id>/', views.send, name='send'),

    path('viewpayment/<int:id>/', views.viewpayment, name='viewpayment'),



]

