from django.contrib import admin
from model.models import Model
# Register your models here.
admin.site.register(Model)
from data.admin import registed

try:
    allmo = Model.objects.all()
    registed(allmo)
except:
    ...