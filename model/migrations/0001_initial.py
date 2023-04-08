# Generated by Django 4.1.7 on 2023-04-08 03:55

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="Model",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(db_index=True, max_length=128, unique=True)),
                ("descriptor", models.TextField()),
                ("ylable", models.CharField(max_length=128)),
                ("woe_iv_table", models.TextField()),
                ("data_type", models.TextField()),
                ("missing_info", models.TextField()),
                ("lr_model_dump", models.BinaryField()),
                ("base_score", models.IntegerField()),
                ("pdo_score", models.IntegerField()),
                ("odds", models.FloatField()),
                ("create_date", models.DateTimeField()),
                ("lable_use", models.TextField()),
                ("rocData", models.TextField()),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
    ]
