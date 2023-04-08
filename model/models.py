from django.db import models
from django.contrib.auth.models import User
# Create your models here.


class Model(models.Model):
    name = models.CharField(max_length=128,db_index=True,unique=True)
    descriptor = models.TextField()
    ylable =  models.CharField(max_length=128)
    woe_iv_table = models.TextField()
    data_type = models.TextField()
    missing_info = models.TextField()
    lr_model_dump = models.BinaryField() #LogisticRegression 序列化数据
    base_score = models.IntegerField()
    pdo_score = models.IntegerField()
    odds = models.FloatField()
    create_date = models.DateTimeField()
    lable_use = models.TextField()
    rocData = models.TextField()
    user = models.ForeignKey(User, on_delete=models.CASCADE)
   
    
  
    
    
    

        
     
       
        
  
    


