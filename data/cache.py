from django.core.cache import cache
from django.core.paginator import Paginator

version = 0

def getPageFromCache(mata):
    
    key = str( (str(mata[0].query),mata[1],mata[2]) )
    ckey = key+'$%^&*count'
    items = []
    count = 0
    if cache.has_key(key,version=version):
        items = cache.get(key,version=version)
        if cache.has_key(ckey,version=version):    
            count = cache.get(ckey,version=version)
        else:
            count = Paginator(mata[0],mata[1]).num_pages
    else:
        paginator = Paginator(mata[0],mata[1])
        items = paginator.page(mata[2])
        count = paginator.count
        items = [i for i in items.object_list]
        if len(items)>0:
            cache.set(key,items,600,version=version)
            cache.set(ckey,count,600,version=version)
        
                       
    return items ,count
    
def deletePageCache():
    global version
    version= version + 1
    
   
    
    
    
    
    