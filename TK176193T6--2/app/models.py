from django.db import models
import os

# Create your models here.
class PatientsModel(models.Model):
    firstname = models.CharField(max_length=100)
    lastname = models.CharField(max_length=100)
    email =  models.EmailField()
    password = models.CharField(max_length=100)
    phone = models.IntegerField()
    address = models.CharField(max_length=100)
    dob = models.DateField()
    profile = models.FileField(upload_to=os.path.join('static', 'patientprofiles'))

    def __str__(self):
        return self.firstname + " " + self.lastname
    
    class Meta:
        db_table = 'PatientsModel'



class DoctorsModel(models.Model):
    firstname = models.CharField(max_length=100)
    lastname = models.CharField(max_length=100)
    email =  models.EmailField()
    password = models.CharField(max_length=100)
    phone = models.IntegerField()
    address = models.CharField(max_length=100)
    designation = models.CharField(max_length=100)
    profile = models.FileField(upload_to=os.path.join('static', 'doctorprofiles'))
    status = models.CharField(max_length=100, default='Pending')

    def __str__(self):
        return self.firstname + " " + self.lastname
    
    class Meta:
        db_table = 'DoctorsModel'


class AppointmentsModel(models.Model):
    patient = models.ForeignKey(PatientsModel, on_delete=models.CASCADE)
    doctor = models.ForeignKey(DoctorsModel, on_delete=models.CASCADE)
    date = models.DateField(null=True)
    disease = models.CharField(max_length=100, null=True)
    symptoms = models.TextField(null=True)
    age =  models.IntegerField(null=True)
    gender = models.CharField(max_length=100,null=True)
    status = models.CharField(max_length=100, default='pending')
    report = models.FileField(upload_to=os.path.join('static', 'reports'),null=True)
    privatekey = models.TextField(null=True)
    publickey = models.TextField(null=True)
    key = models.IntegerField(null=True)

    class Meta:
        db_table = 'AppointmentsModel'


class FeedbackModel(models.Model):
    patient = models.ForeignKey(PatientsModel, on_delete=models.CASCADE)
    feedback = models.TextField()


    def __str__(self):
        return self.patient.firstname  + " " + self.patient.lastname
    
    class Meta:
        db_table = 'FeedbackModel'


class FreeQuote(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    phone = models.IntegerField()
    note = models.TextField()

    def __str__(self):
        return self.name
    
    class Meta:
        db_table = 'FreeQuoteModel'


from django.db import models

class Specialist(models.Model):
    specialization = models.CharField(max_length=255, unique=True)
    
    def __str__(self):
        return self.specialization

    class Meta:
        db_table = 'specialists'


class Disease(models.Model):
    name = models.CharField(max_length=255, unique=True)
    specialists = models.ManyToManyField(Specialist, related_name="diseases_handled")
    
    def __str__(self):
        return self.name
    
    class Meta:
        db_table = 'diseases'



class Payment(models.Model):
    appointment = models.ForeignKey(AppointmentsModel,on_delete=models.CASCADE)
    username = models.CharField(max_length=100)
    cardnumber = models.IntegerField()
    cvv = models.IntegerField()
    expirydate = models.CharField(max_length=100)
    amount = models.IntegerField()
    status = models.CharField(max_length=100,default='Payment Successfull')

    def __str__(self):
        return self.username
    class Meta:
        db_table = 'Payment'

        

