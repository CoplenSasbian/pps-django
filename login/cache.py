from django.core.cache import cache
from django.contrib.auth.models import User
def del_cache_by_id(id: int):
    cache.delete('user.{}'.format(id))


def get_user_by_id(id:int):
    key = 'user.{}'.format(id)
    user = cache.get(key)
    if not user:
       user = User.objects.get(id=id)
       cache.set(key,user)
    return user