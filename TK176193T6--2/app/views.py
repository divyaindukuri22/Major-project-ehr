from django.shortcuts import render, redirect
from django.contrib import messages
from . models import *
from django.http import JsonResponse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import re
from django.core.paginator import Paginator
from django.http import HttpResponse

from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from . final_2 import *

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.concatkdf import ConcatKDFHash
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import os

from Crypto.Cipher import AES

# Create your views here.
def index(request):
    data=FeedbackModel.objects.all()
    return render(request, 'index.html',{'data': data})


def about(request):
    return render(request, 'about.html')

def patientregister(request):
    if request.method == 'POST':
        firstname  = request.POST['firstname']
        lastname = request.POST['lastname']
        email = request.POST['email']
        dob = request.POST['dob']
        address = request.POST['address']
        phone = request.POST['phone']
        profile = request.FILES['image']
        password = request.POST['password']
        confirmpassword = request.POST['confirmpassword']
        if password == confirmpassword:
            patient = PatientsModel.objects.filter(email=email).exists()
            if patient:
                messages.error(request, 'Email already exists')
                return redirect('patientregister')
            else:
                patient = PatientsModel(firstname=firstname, lastname=lastname, email=email, dob=dob,
                                        address=address, phone=phone, profile=profile, password = password)
                patient.save()
                messages.success(request, 'Your Registration is Successfull!')
                return redirect('patientregister')
        else:
            messages.error(request, 'Password and Confirm Password does not match')
            return redirect('patientregister')
    return render(request, 'patientregister.html')

def patientlogin(request):
    # PatientsModel.objects.all().delete()
    # DoctorsModel.objects.all().delete()
    # AppointmentsModel.objects.all().delete()
    if request.method == 'POST':
        
        email = request.POST['email']
        
        password = request.POST['password']
        patient = PatientsModel.objects.filter(email=email, password=password).exists()
        if patient:
            data = PatientsModel.objects.get(email=email)
            request.session['login']='patient'
            request.session['email']=email
            request.session['name']=data.firstname+" "+data.lastname
            return redirect('home')
        else:
            messages.error(request, 'Invalid email or password!')
            return redirect('patientlogin')
    return render(request, 'patientlogin.html')

def doctorregister(request):
    if request.method == 'POST':
        firstname  = request.POST['firstname']
        lastname = request.POST['lastname']
        email = request.POST['email']
        designation = request.POST['designation']
        address = request.POST['address']
        phone = request.POST['phone']
        profile = request.FILES['image']
        password = request.POST['password']
        confirmpassword = request.POST['confirmpassword']
        if password == confirmpassword:
            patient = DoctorsModel.objects.filter(email=email).exists()
            if patient:
                messages.error(request, 'Email already exists')
                return redirect('doctorregister')
            else:
                patient = DoctorsModel(firstname=firstname, lastname=lastname, email=email, designation=designation,
                                        address=address, phone=phone, profile=profile, password = password)
                patient.save()
                messages.success(request, 'Your Registration is Successfull!')
                return redirect('doctorregister')
        else:
            messages.error(request, 'Password and Confirm Password does not match')
            return redirect('doctorregister')
    return render(request, 'doctorregister.html')

def doctorlogin(request):

  
    if request.method == 'POST':
    
        email = request.POST['email']
        
        password = request.POST['password']
        doctor = DoctorsModel.objects.filter(email=email, password=password).exists()
        if doctor:
            if DoctorsModel.objects.filter(email=email, status='Authorized').exists():
                data = DoctorsModel.objects.get(email=email)
                request.session['login']='doctor'
                request.session['email']=email
                request.session['name']=data.firstname+" "+data.lastname
                return redirect('home')
            else:
                messages.error(request, 'Your account is not authorized')
                return redirect('doctorlogin')
        else:
            messages.error(request, 'Invalid email or password!')
            return redirect('doctorlogin')
    return render(request, 'doctorlogin.html')


def adminlogin(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']
        # print('==================================================')
        # print(email,password)
        if email == 'admin@gmail.com' and password == 'admin':
            request.session['login']='admin'
            request.session['email']=email
            request.session['name']='admin'
            return redirect('home')
        else:
            messages.error(request, 'Invalid email or password!')
            return redirect('adminlogin')
    return render(request, 'adminlogin.html')


def viewdoctors(request):
    login = request.session['login']
    data = DoctorsModel.objects.all()
    paginator = Paginator(data, 4)
    page_number = request.GET.get('page')
    page_data = paginator.get_page(page_number)
    return render(request, 'viewdoctors.html', {'data':page_data,'login': login})

def authorize(request, id):
    data = DoctorsModel.objects.get(id=id)
    data.status = 'Authorized'
    data.save()
    messages.success(request, f'{data.firstname} {data.lastname} Authorized Successfully!')
    return redirect('viewdoctors')

def appointments(request):
    login = request.session['login']
    data = AppointmentsModel.objects.all()
    paginator = Paginator(data, 4)
    page_number = request.GET.get('page')
    page_data = paginator.get_page(page_number)
    return render(request, 'appointments.html',{'data':page_data,'login': login})




def home(request):
    # PatientsModel.objects.all().delete()
    # DoctorsModel.objects.all().delete()
    # AppointmentsModel.objects.all().delete()
    login =  request.session['login']
    data=FeedbackModel.objects.all()
    return render(request, 'home.html',{'login':login,'data': data})


def logout(request):
    del request.session['login']
    del request.session['email']
    del request.session['name']
    return redirect('index')



import time
@csrf_exempt
def chatbot_response(request):
    login = request.session.get('login', False)
    name = request.session.get('name', 'Guest')

    # Function to extract symptoms with negation handling
    # Function to extract symptoms with negation handling and variations
    def extract_symptoms(text, symptoms_map):
        symptoms_dict = {symptom: 0 for symptom in symptoms_map.keys()}  # Default all to 0 (no)
        
        # Convert text to lowercase
        text = text.lower()

        # Handle negations and variations using regex
        for symptom, variations in symptoms_map.items():
            # Check if any variation is negated
            negation_pattern = r'\b(no|not|don\'t|doesn\'t|haven\'t|isn\'t|aren\'t) (have )?(' + '|'.join(map(re.escape, variations)) + r')\b'
            positive_pattern = r'\b(' + '|'.join(map(re.escape, variations)) + r')\b'
            
            if re.search(negation_pattern, text):
                symptoms_dict[symptom] = 0  # If negated, symptom is set to 0
            elif re.search(positive_pattern, text):
                symptoms_dict[symptom] = 1  # Set to 1 (yes) if the symptom is mentioned positively

        # Check if any symptoms were detected positively
        found_symptom = any(symptoms_dict.values())
        
        if not found_symptom:
            print("No symptoms found in the input. Please provide relevant symptoms.")

        return list(symptoms_dict.values()), found_symptom

    if request.method == "POST":
        print("POST request received.")  # Debugging line

        # Check if it's an image upload
        if 'image' in request.FILES:
            file = request.FILES['image']  
            image_path = os.path.join(settings.BASE_DIR, 'static', 'classification', file.name)
    
            # Ensure the classification folder exists
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            
            # Save the file to the specified location
            with open(image_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
       
            # Example usage
            # image_path = r'data\train\no_tumor\231_jpg.rf.18afbbe365e4becf09170ac6e3edccbb.jpg'  # Change to the path of your image

            # Step 1: Check if the image is relevant
            relevance_prediction = predict_relevance(image_path)
            if relevance_prediction == 1:  # Irrelevant image
                
                response ={
                        'Prediction': "Image is irrelevant. Please provide relevent brain MRI image."
                    }
                return JsonResponse(response)
            else:  # Relevant image, proceed with tumor prediction
                print("Image is relevant. Proceeding with tumor detection...")
                
                # Step 2: Classify the image for tumor presence
                tumor_prediction = predict_image(image_path)
                

                # Map prediction to label
                if tumor_prediction == 1:  # Tumor detected
                    print("Tumor detected. Proceeding to segmentation...")
                    predicted_mask = predict_segmentation(image_path)

                    # Visualize the input image and predicted mask
                    # plt.figure(figsize=(10, 5))
                    # plt.subplot(1, 2, 1)
                    # plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
                    # plt.title('Input Image')
                    # plt.axis('off')

                    # plt.subplot(1, 2, 2)
                    # plt.imshow(predicted_mask, cmap='gray')
                    # plt.title('Predicted Mask')
                    # plt.axis('off')

                    # plt.show()
                    predicted_mask = (predicted_mask * 255).astype(np.uint8)

            # Create an image from the NumPy array
                    image = Image.fromarray(predicted_mask)
                    timestamp = str(int(time.time()))
                    image_filename = f'predicted_mask_{timestamp}.png'

                    # Set the path where the image will be saved with the new filename
                    image_path = os.path.join(settings.BASE_DIR, 'static', 'predictedmask', image_filename)

                    # Set the path where the image will be saved
                    # image_path = os.path.join(settings.BASE_DIR, 'static', 'predictedmask', 'predicted_mask.png')
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    image.save(image_path)


                    # Generate the relative URL path based on the image path
                    relative_image_path = os.path.relpath(image_path, start=settings.BASE_DIR)

                    # Ensure the relative path starts with /static/
                    relative_image_path = '/' + relative_image_path.replace('\\', '/') 
                    request.session['image'] = relative_image_path
        
                    # os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    # with open(image_path, 'wb+') as destination:
                    #     for chunk in file.chunks():
                    #         destination.write(chunk)
                
                    response ={
                        'Prediction': "Tumor detected.",
                        'img' : relative_image_path
                    }
                    print('=====================================================')
                    print(relative_image_path)
                    return JsonResponse(response)
                else:  # No tumor detected
                    print("No tumor detected. No segmentation performed.")
                    response ={
                        'Prediction': "No Tumor detected."
                    }
                    return JsonResponse(response)

            # Otherwise, handle the text input from request.POST
        message = request.POST.get('message', '')
        print(f"Message received: {message}")

        # Add bot response logic
        if message:
            request.session['symp'] = message
            # Load the training dataset
            df = pd.read_csv('static/dataset/Training.csv')

            # Print column names to check what's available
            print("Columns in the dataset:", df.columns)

            # Drop unnecessary unnamed columns
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')] 
            # Define the symptom variations mapping based on the columns used for training
            symptoms_map = {
                'itching': ['itching', 'skin itching', 'itchy', 'itch', 'irritation', 'pruritus', 'skin irritation'],
                'skin_rash': ['skin rash', 'rashes', 'rash', 'red spots', 'blotchy skin', 'skin irritation', 'dermatitis', 'skin redness'],
                'nodal_skin_eruptions': ['nodal skin eruptions', 'skin bumps', 'skin eruptions', 'nodules', 'skin nodules'],
                'continuous_sneezing': ['continuous sneezing', 'frequent sneezing', 'sneezing fits', 'uncontrollable sneezing', 'repeated sneezing', 'sneeze attacks'],
                'chills': ['chills', 'shivers', 'cold sensation', 'feeling cold', 'fever chills', 'cold tremors'],
                'joint_pain': ['back pain','joint pain', 'joint discomfort', 'arthritis pain', 'joint ache', 'stiff joints', 'aching joints', 'sore joints', 'arthralgia'],
                'stomach_pain': ['stomach pain', 'abdominal pain', 'belly ache', 'tummy ache', 'gas pain', 'cramps', 'stomach ache', 'intestinal pain'],
                'acidity': ['acidity', 'heartburn', 'acid reflux', 'indigestion', 'gastric pain', 'acid stomach', 'stomach acid'],
                'ulcers_on_tongue': ['ulcers on tongue', 'tongue ulcers', 'mouth sores', 'canker sores', 'painful sores', 'mouth ulcers'],
                'muscle_wasting': ['muscle wasting', 'muscle loss', 'muscle degeneration', 'loss of muscle mass', 'muscle weakness', 'muscular atrophy'],
                'vomiting': ['vomiting', 'throwing up', 'nausea', 'retching', 'puking', 'sick to the stomach', 'upchucking', 'regurgitation', 'emesis'],
                'burning_micturition': ['burning micturition', 'painful urination', 'burning sensation while urinating', 'dysuria', 'burning urine', 'pain while peeing'],
                'spotting urination': ['spotting urination', 'frequent urination', 'urinary incontinence', 'difficulty urinating', 'trouble urinating', 'leaking urine'],
                'fatigue': ['fatigue', 'tiredness', 'exhaustion', 'weariness', 'low energy', 'weakness', 'lethargy', 'feeling drained', 'sluggishness'],
                'weight_gain': ['weight gain', 'increased weight', 'putting on weight', 'gaining weight', 'unexplained weight gain', 'excess weight'],
                'anxiety': ['anxiety', 'nervousness', 'unease', 'worry', 'stress', 'fear', 'panic', 'feeling anxious'],
                'cold_hands_and_feets': ['cold hands and feet', 'cold extremities', 'poor circulation', 'cold fingers', 'cold toes', 'numb hands', 'numb feet']
            }


            # Prepare the list of symptoms for input
            symptoms_list = list(symptoms_map.keys())

            # Extract symptoms based on the user input
            newdata, found_symptom = extract_symptoms(message, symptoms_map)

            # If no symptoms were found, don't proceed with prediction
            if not found_symptom:
                response={
                    'Prediction' : "No prediction can be made without relevant symptoms."
                }
                print(response)
                return JsonResponse(response)
                
            else:
                # Extract the feature names from the training data (excluding 'prognosis')
                training_features = list(df.columns[:-1])  # All except 'prognosis'

                # Initialize the input data with 0s for all features
                final_input_data = [0] * len(training_features)
                
                # Map the extracted symptoms to the corresponding features in the training data
                for i, feature in enumerate(training_features):
                    if feature in symptoms_list:
                        final_input_data[i] = newdata[symptoms_list.index(feature)]

                # Check if the number of features match before proceeding
                if len(final_input_data) == len(training_features):
                    # Filter the DataFrame using available columns
                    df = df[training_features + ['prognosis']]  # Ensure 'prognosis' is included

                    # Prepare data for training
                    x = df.drop(['prognosis'], axis=1)
                    y = df['prognosis']
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

                    # Train the Random Forest model
                    rf = RandomForestClassifier()
                    rf.fit(x_train, y_train)

                    # Make prediction based on the final input data
                    result = rf.predict([final_input_data])
                    result = result[0]
                    
                    request.session['disease'] = result
                    response = {
                    'Prediction' : f'The predicted disease is {result}' ,
                       "Link": {
                    "label": "If you want Appointment with the Doctor?, Click Here",
                    "url": f"http://127.0.0.1:8000/doctors/{result}/"
                       }
                }
                    print(response)
                else:
                    response ={
                    
                          'Prediction' : "Feature mismatch! Ensure the input data matches the trained model features."
                    }

                print(f"Response: {response}")  # Debugging line
                return JsonResponse(response)

    # If the request is not a POST, render the chatbot template
    return render(request, 'chatbot.html', {'login': login, 'name': name})


def doctors(request, disease):

    # print(pred)
    # disease = request.session['disease']
    login = request.session['login']
    name = request.session['name']
    email = request.session['email']
    patient = PatientsModel.objects.get(email=email)
    print(patient.address)
    # dd = DoctorsModel.objects.get(address = patient.address)
    # print(dd)
    dis = Disease.objects.get(name=disease)
    # print(dis,"==================================================")
    specialists = dis.specialists.all()    
    # for specialist in specialists:
    #     print(specialist.specialization, "=======================================")
    designation = specialists.first().specialization
    # print(designation,"=======================================")
    # print(patient.address)
    # patient_address = patient.address.strip() 
    data = DoctorsModel.objects.filter(address=patient.address,designation=designation)
    dat = DoctorsModel.objects.all()
    # for i in dat:
    #     print(i.address,"=========nnnnnnnnnnnnnnnnnnnnnnnnnn=================")
    # print("data",data)
    if data:
        paginator = Paginator(data, 6)
        page_number = request.GET.get('page')
        page_data = paginator.get_page(page_number)
        return render(request, 'doctors.html', {'data':page_data,'login': login, 'name': name}) 
    else:
        
        messages.success(request, 'No Doctors Available at your Loactaion')
        return render(request, 'doctors.html', {'login': login, 'name': name})


def contact(request):
    return render(request, 'contact.html')


def profile(request):

    # AppointmentsModel.objects.all().delete()
    login = request.session['login']
    name = request.session['name']
    email = request.session['email']
    if login == 'patient':
        data =  PatientsModel.objects.filter(email=email)
        return render(request, 'userprofile.html',{'login': login, 'name': name, 'data':data})
    else:
        data =  DoctorsModel.objects.filter(email=email)
        return render(request, 'doctorprofile.html',{'login': login, 'name': name, 'data':data})
    


def editprofile(request):
    login = request.session['login']
    email = request.session['email']
    if login == 'patient':
        data = PatientsModel.objects.filter(email=email)
    else:
        data = DoctorsModel.objects.filter(email=email)
    if request.method == 'POST':
        
        phone = request.POST['phone']
        address = request.POST['address']
      
        if login == 'patient':
            user = PatientsModel.objects.get(email=email)
            if 'profile' in request.FILES:
                profile = request.FILES['profile']
                user.phone = phone
                user.address = address
                user.profile = profile
                user.save()
            else:
                user.phone = phone
                user.address = address
                user.save()
        else:
            user = DoctorsModel.objects.get(email=email)
            if 'profile' in request.FILES:
                profile = request.FILES['profile']
                user.phone = phone
                user.address = address
                user.profile = profile
                user.save()
            else:
                user.phone = phone
                user.address = address
                user.save()  

        messages.success(request, 'Profile Updated Successfully!')
        return redirect('profile')

    return render(request, 'editprofile.html',{'login':login,'data':data})

def send(request, id):
    data = AppointmentsModel.objects.get(id=id)
    data.status = 'Forwarded To Doctor'
    data.save()
    doctor = DoctorsModel.objects.get(id=data.doctor.id)
    patient = PatientsModel.objects.get(id=data.patient.id)
    # Email details for the doctor
    email_subject = 'New Appointment Request - Please Schedule Appointment Time'
    email_message = f'''
    Hello Dr. {doctor.firstname} {doctor.lastname},

    A new patient appointment request has been made. Please review the details below and schedule the appointment time.

    - Patient First Name: {patient.firstname}
    - Patient Last Name: {patient.lastname}
    - Patient Email: {patient.email}
    -Disease: {data.disease}
    - Symptoms: {data.symptoms}
    - Age: {data.age}
    - Gender: {data.gender}

    Once you confirm the appointment date, we will send the patient the final details.

    Please reply with the appointment date, or if there are any issues with this request.

    Best regards,
    Your Website Team
    '''

    # Send the email to the doctor
    send_mail(email_subject, email_message, 'cse.takeoff@gmail.com', [doctor.email])
    
    # Sample email details
    email_subject = 'Your Appointment is Scheduled'
    email_message = f'''
    Hello {patient.firstname},

    Your appointment has been successfully scheduled with Dr. {doctor.firstname} {doctor.lastname}. Here are the details of your appointment:

    - First Name: {patient.firstname}
    - Last Name: {patient.lastname}
    - Email: {patient.email}
    - Doctor: Dr. {doctor.firstname} {doctor.lastname}
    - Disease: {data.disease}
    - Symptoms: {data.symptoms}
    - Age: {data.age}
    - Gender: {data.gender}

    Please arrive at the scheduled time. We look forward to assisting you with your health needs.

    Best regards,
    Your Website Team
    '''

    # Send the email
    send_mail(email_subject, email_message, 'cse.takeoff@gmail.com', [patient.email])
  # Assuming the doctor's email is saved as data.appointment.doctor.email
    messages.success(request, 'Appointment mail sent successfully!')
    return redirect('appointments')


def viewpayment(request, id):
    # Assuming the payment details are saved in the database
    data = AppointmentsModel.objects.get(id=id)
    age = data.age
    gender = data.gender
    print("jhcbsadcasbd",gender)
    payments = Payment.objects.filter(appointment=data)

    doctor= DoctorsModel.objects.filter(id=data.doctor.id)
    patient = PatientsModel.objects.filter(id=data.patient.id)
    return render(request, 'payment.html', {'age': age, 'gender': gender, 'payment': payments,'doctor':doctor,'data':patient})


def sendappointment(request, id):
    login = request.session['login']
    email = request.session['email']
    symp = request.session['symp']
    disease = request.session['disease']
    doctor= DoctorsModel.objects.filter(id=id)
    patient = PatientsModel.objects.filter(email=email)
    if request.method == "POST":
        age = request. POST['age']
        gender = request.POST['gender']
        cardnumber = request.POST['cardnumber']
        cardholder = request.POST['cardholder']

        cvv = request.POST['cvv']
        expdate = request.POST['expdate']
        amount = request.POST['amount']


        doctor = DoctorsModel.objects.get(id=id)
        patient = PatientsModel.objects.get(email=email)
        appointment = AppointmentsModel.objects.create(
            doctor = doctor,
            patient = patient,
            disease = disease,
            symptoms = symp,
            age = age,
            gender = gender
        )
        appointment.save()
        appreq = AppointmentsModel.objects.get(id=appointment.id)
        Payment.objects.create(
            appointment = appreq,
            username = cardholder,
            cardnumber = cardnumber,
            cvv = cvv,
            expirydate = expdate,
            amount = amount
        ).save()
        

        messages.success(request, 'Appointment Request Sent Successfully!')
        return redirect('myappointments') 
    return render(request, 'appointment.html',{'login':login,'doctor':doctor,'data':patient, 'id':id})


def myappointments(request):
    login = request.session['login']
    email = request.session['email']
    name =  request.session['name']
    appointments = AppointmentsModel.objects.filter(patient__email = email)
    paginator = Paginator(appointments, 6)
    page_number = request.GET.get('page')
    page_data = paginator.get_page(page_number)
    return render(request, 'myappointments.html', {'data':page_data,'login': login, 'name': name})


def withdraw(request, id):
    login = request.session['login']
    email = request.session['email']
    appointment = AppointmentsModel.objects.get(id=id)
    appointment.delete()
    messages.success(request, 'Appointment Cancelled Successfully!')
    return redirect('withdraw', id)


def viewappointments(request):
    login = request.session['login']
    email = request.session['email']
    name =  request.session['name']
    appointments = AppointmentsModel.objects.filter(Q(doctor__email = email) &
                                                    ~Q(status = 'pending'))
                                                    
    paginator = Paginator(appointments, 6)
    page_number = request.GET.get('page')
    page_data = paginator.get_page(page_number)
    return render(request, 'viewappointments.html', {'data':page_data,'login': login, 'name': name})


def accept(request, id):
    login = request.session['login']
    email = request.session['email']
    if request.method == 'POST':
        data = AppointmentsModel.objects.get(id=id)
        data.date = request.POST['date']
        data.status = 'Accepted'
        data.save()
        doctor = DoctorsModel.objects.get(id=data.doctor.id)
        patient = PatientsModel.objects.get(id=data.patient.id)
         # Sample email details
        email_subject = 'Your Appointment is Scheduled'
        email_message = f'''
        Hello {patient.firstname},

        Your appointment has been successfully scheduled with Dr. {doctor.firstname} {doctor.lastname}. Here are the details of your appointment:

        - First Name: {patient.firstname}
        - Last Name: {patient.lastname}
        - Email: {patient.email}
        - Doctor: Dr. {doctor.firstname} {doctor.lastname}
        - Appointment Date: {data.date}
        - Disease: {data.disease}
        - Symptoms: {data.symptoms}
        - Age: {data.age}
        - Gender: {data.gender}

        Please arrive at the scheduled time. We look forward to assisting you with your health needs.

        Best regards,
        Your Website Team
        '''

        # Send the email
        send_mail(email_subject, email_message, 'cse.takeoff@gmail.com', [patient.email])
        messages.success(request, 'Appointment Accepted Successfully!')
        return redirect('viewappointments')
    

import os
from django.http import HttpResponse, Http404
from django.conf import settings

def downloadmri(request):
    # Retrieve the relative image path from session (e.g., '/static/predictedmask/predicted_mask_1729660799.png')
    image_path = request.session.get('image')

    if not image_path:
        # Return error if no image path is stored in session
        return HttpResponse("No image found in session.", status=400)

    # Remove the '/static/' prefix and convert the path to an absolute file path
    static_root = os.path.join(settings.BASE_DIR, 'static')  # Path to your static directory
    relative_image_path = image_path.replace('/static/', '')  # Remove the '/static/' prefix
    absolute_image_path = os.path.join(static_root, relative_image_path)  # Full path to the image

    # Log the paths for debugging
    print(f"Relative image path: {relative_image_path}")
    print(f"Absolute image path: {absolute_image_path}")

    # Check if the file exists on disk
    if not os.path.exists(absolute_image_path):
        raise Http404("File not found")

    # Open the file and return it as a downloadable response
    try:
        with open(absolute_image_path, 'rb') as image_file:
            response = HttpResponse(image_file.read(), content_type='image/png')
            response['Content-Disposition'] = 'attachment; filename="predicted_mask.png"'
            return response
    except Exception as e:
        # Handle unexpected errors during file reading
        return HttpResponse(f"An error occurred: {str(e)}", status=500)
    


def viewreports(request):
    
    login = request.session['login']
    email = request.session['email']
    name =  request.session['name']
    
    appointments = AppointmentsModel.objects.filter(doctor__email = email, status='Accepted')

    paginator = Paginator(appointments, 6)
    page_number = request.GET.get('page')
    page_data = paginator.get_page(page_number)
    return render(request, 'viewreports.html', {'data':page_data,'login': login, 'name': name})


def report(request, id):
    
    login = request.session['login']
    email = request.session['email']
    name =  request.session['name']
    appointment = AppointmentsModel.objects.filter(id=id)
    return render(request, 'reports.html', {'data':appointment,'login': login, 'name':name, 'id':id})






# AES Encryption and Decryption Functions
def aes_encrypt(key, data):
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(data, AES.block_size))
    return cipher.iv + ct_bytes



# ECC Key Exchange Functions
def ecc_key_pair():
    private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
    public_key = private_key.public_key()
    return private_key, public_key

def ecc_shared_secret(private_key, peer_public_key):
    shared_secret = private_key.exchange(ec.ECDH(), peer_public_key)
    # Derive AES key using ConcatKDF
    ckdf = ConcatKDFHash(algorithm=hashes.SHA256(), length=16, otherinfo=None, backend=default_backend())
    aes_key = ckdf.derive(shared_secret)
    return aes_key

# Serialization and Deserialization for Public Key
def serialize_public_key(public_key):
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

def deserialize_public_key(public_bytes):
    return serialization.load_pem_public_key(public_bytes, backend=default_backend())

# Encryption Function
def hybrid_encrypt(data, sender_private_key, receiver_public_key):
    # Derive shared secret using ECC
    shared_secret = ecc_shared_secret(sender_private_key, receiver_public_key)
    
    # Encrypt data using AES with the shared secret as the key
    encrypted_data = aes_encrypt(shared_secret, data.encode('utf-8'))
    
    return encrypted_data





import random
from cryptography.hazmat.primitives import serialization

def uploadreport(request, id):
    login = request.session.get('login')
    
    if request.method == 'POST' and request.FILES.get('file'):
        report = request.FILES['file']

        # Fetch the appointment data
        try:
            data = AppointmentsModel.objects.get(id=id)
        except AppointmentsModel.DoesNotExist:
            return HttpResponse("Appointment not found", status=404)

        # Define the path where the file will be saved
        temp_file_path = os.path.join('static', 'reports', report.name)
        with open(temp_file_path, 'wb+') as destination:
            for chunk in report.chunks():
                destination.write(chunk)

        # Read the file content as binary
        with open(temp_file_path, 'r') as f:
            file_content = f.read()

        # Generate ECC key pairs for sender and receiver
        sender_private_key, sender_public_key = ecc_key_pair()
        receiver_private_key, receiver_public_key = ecc_key_pair()

        # Serialize the receiver's private key and sender's public key to save in the model
        receiver_private_key_pem = receiver_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')

        sender_public_key_pem = sender_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')

        # Encrypt the file content
        encrypted_message = hybrid_encrypt(file_content, sender_private_key, receiver_public_key)

        # Write the encrypted content to the file
        with open(temp_file_path, 'wb') as destination:
            destination.write(encrypted_message)

        # Save report details in the database
        data.report = temp_file_path
        data.privatekey = receiver_private_key_pem  # Store the receiver's private key as PEM
        data.publickey = sender_public_key_pem  # Store the sender's public key as PEM
        data.status = 'Report Uploaded'
        data.key = random.randint(111111, 999999)
        data.save()
        
        messages.success(request, 'Report Uploaded Successfully!')

        return redirect('viewreports')

    return HttpResponse("Invalid request", status=400)


from django.db.models import Q
def myreports(request):
    # data = AppointmentsModel.objects.get(id=10)
    # data.report=''
    # data.privatekey=''
    # data.publickey=''
    # data.status='Report is Ready to Download'
    # data.key=0
    # data.save()
    login = request.session['login']
    email =  request.session['email']
    name =  request.session['name']
    appointments = AppointmentsModel.objects.filter(
        Q(patient__email=email) & 
        (Q(status='Report Uploaded') | Q(status='Report is Ready to Download'))
    ) 
    paginator = Paginator(appointments, 6)
    page_number = request.GET.get('page')
    page_data = paginator.get_page(page_number)
    return render(request, 'myreports.html', {'data':page_data,'login': login, 'name': name})

def aes_decrypt(key, enc_data):
    iv = enc_data[:AES.block_size]
    ct = enc_data[AES.block_size:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    pt = unpad(cipher.decrypt(ct), AES.block_size)
    return pt


# Decryption Function
def hybrid_decrypt(encrypted_data, receiver_private_key, sender_public_key):
    # Derive shared secret using ECC
    shared_secret = ecc_shared_secret(receiver_private_key, sender_public_key)
    
    # Decrypt data using AES with the shared secret as the key
    decrypted_data = aes_decrypt(shared_secret, encrypted_data)
    
    return decrypted_data.decode('utf-8')

from django.core.mail import send_mail
from cryptography.hazmat.primitives import serialization

def getreport(request, id):
    login = request.session['login']
    email = request.session['email']
    name = request.session['name']
    dat = AppointmentsModel.objects.filter(id=id)  
    # Fetch the appointment data
    try:
        data = AppointmentsModel.objects.get(id=id)
    except AppointmentsModel.DoesNotExist:
        return HttpResponse("Appointment not found", status=404)
    if  data.status == "Report is Ready to Download":
        return render(request, 'getreport.html', {'login': login, 'name': name, 'id': id, 'data': dat})
    else:
        # Deserialize the receiver's private key
        receiver_private_key = serialization.load_pem_private_key(
            data.privatekey.encode('utf-8'),  # Convert PEM string back to bytes
            password=None,  # Use password if the private key is encrypted
            backend=default_backend()
        )

        # Deserialize the sender's public key
        sender_public_key = deserialize_public_key(data.publickey.encode('utf-8'))

        # Read the encrypted report file content
        with open(data.report.path, 'rb') as file:
            file_content = file.read()

        # Decrypt the file content using ECC and AES
        decrypted_message = hybrid_decrypt(file_content, receiver_private_key, sender_public_key)

        # Write the decrypted content back to the report file (if needed)
        with open(data.report.path, 'wb') as files:
            files.write(decrypted_message.encode('utf-8'))
        data.status = "Report is Ready to Download"
        data.save()
        email_subject = 'Your Key Details'
        email_message = f'Hello {data.patient.firstname}\n\nWelcome To Our Website!\n\nHere are your Key Details:\nFirst Name: {data.patient.firstname}\nLast Name: {data.patient.lastname}\nEmail: {data.patient.email}\nReport Status: Uploaded \nKey: {data.key}\n\nPlease keep this information safe.\n\nBest regards,\nYour Website Team'
        send_mail(email_subject, email_message, 'cse.takeoff@gmail.com', [data.patient.email])
       
        
        # Render the report page
        return render(request, 'getreport.html', {'login': login, 'name': name, 'id': id, 'data': dat})


from django.http import FileResponse

def download(request, id):
    login = request.session['login']
    email = request.session['email']
    context = AppointmentsModel.objects.get(id=id)
    print(context.key)
    if request.method == 'POST':
        key = request.POST['key']
        # print(type(key), context.key)
        if int(key) == int(context.key):
            file_path = context.report.path  # Get the file path
            file_name = context.report.name.split('/')[-1]  # Extract the file name
            response = FileResponse(open(file_path, 'rb'), as_attachment=True, filename=file_name)
            return response
        else:
            messages.success(request, 'You Entered key is Wrong')
            return redirect('getreport', id)
        

def feedback(request):
    login = request.session['login']
    email = request.session['email']
    data =  PatientsModel.objects.filter(email=email)
    if request.method == 'POST':
        feedback = request.POST['feedback']
        dat =  PatientsModel.objects.get(email=email)
        feed = FeedbackModel.objects.create(
            patient=dat,
            feedback=feedback
        )
        feed.save()
        messages.success(request, 'Your Feedback is Successfully Sent')
    return render(request, 'feedback.html',{'login':login, 'data':data})


def freequote(request):
    if request.method == 'POST':
        name = request.POST['name']
        email = request.POST['email']
        phone = request.POST['phone']
        message = request.POST['message']
        quote = FreeQuote.objects.create(
            name=name,
            email=email,
            phone=phone,
            note=message
        )
        quote.save()
        messages.success(request, 'Your Free Quote is Successfully Sent')
        login = request.session.get('login', None)

        if login:
            return redirect('home')
        else:
            return redirect('index')