from django.contrib import admin

# Register your models here.
    
from card_model.v2_model import ModelPersistence
from data.models import getModelFormDataType


def registed(allMo):
    try:
        for i in allMo:
            mo = ModelPersistence(i)
            mo = getModelFormDataType(mo.getName(),mo.getDataType(),mo.getLableUse())
            admin.site.register(mo)
    except :
        ...
        
