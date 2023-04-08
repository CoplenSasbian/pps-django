from django.core.cache import cache
from .models import Model
def get_model_by_id(id:int):
    key = 'model.{}'.format(id)
    data = cache.get(key)
    if not data:
        data =  Model.objects.get(id=id)
        cache.set(key,data,3600)
    return data

def get_all():
    all = cache.get('model.all')
    if not all:
        all = Model.objects.all()
        cache.set('model.all',all,3600)
    return all