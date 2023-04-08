from django.urls import path
from . import views
urlpatterns =[
    path('page',views.getPage),
    path('tojson',views.readToJSON),
    path('adddatas',views.addDatas),
    path('pridict',views.pridictData),
    path('coefficient',views.getCoefficient),
    path('intercept',views.getIntercept),
]