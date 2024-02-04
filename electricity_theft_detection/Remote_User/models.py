from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)


class theft_detection_type(models.Model):

    LCLid= models.CharField(max_length=3000)
    Age= models.CharField(max_length=3000)
    Sex= models.CharField(max_length=3000)
    day= models.CharField(max_length=3000)
    energy_median= models.CharField(max_length=3000)
    energy_mean= models.CharField(max_length=3000)
    energy_max= models.CharField(max_length=3000)
    energy_count= models.CharField(max_length=3000)
    energy_std= models.CharField(max_length=3000)
    energy_sum= models.CharField(max_length=3000)
    energy_min= models.CharField(max_length=3000)
    Prediction= models.CharField(max_length=3000)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



