from django.db import models

# Create your models here.

from card_model.v2_model import ModelPersistence
def _create_model(name, fiedls=None, app_lable="", module="", option=None):
    class Meta:
        pass
    if app_lable:
        setattr(Meta, "app_lable", app_lable)

    if option is not None:
        for key, value in option.items():
            setattr(Meta, key, value)
    attrs = {"__module__": module, "Meta": Meta}
    if fiedls:
        attrs.update(fiedls)
    return type(name, (models.Model,), attrs)


def _create_database_table(model):
    from django.db import connection
    from django.db.backends.base.schema import BaseDatabaseSchemaEditor

    editor = BaseDatabaseSchemaEditor(connection)
    try:
        editor.create_model(model)
        return None
    except AttributeError as err:
        return err


def _pdtype2dbtype(type: str,null = True):
    if type == 'string':
        return models.CharField(max_length=256,null = null)
    elif type == 'integer':
        return models.IntegerField(null = null)
    elif type == 'floating':
        return models.FloatField(null = null)  


def getModelFormDataType(mo:ModelPersistence):
    types = mo.getDataType()
    ylable = mo.getYLable()
    lableUse = mo.getLableUse()
    name = mo.getName()
    miss = mo.getMissingInfo()
    fileds = {}
    for i, v in types.items():
        if i == ylable:continue
        null = not lableUse[i] or (miss[i]['missingMethod']=='自成分箱' and  miss[i]['nullCount']>0)
    
        
        fileds[i] = _pdtype2dbtype(v,null)
    return _create_model(name, fileds, "", "data.models")


def createModelFormDataType(mo:ModelPersistence):
    model = getModelFormDataType(mo)
    _create_database_table(model)
    return model




    
