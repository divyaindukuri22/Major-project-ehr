# Generated by Django 5.1.2 on 2024-10-21 11:59

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0005_chatmessage_image'),
    ]

    operations = [
        migrations.CreateModel(
            name='AppointmentsModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField()),
                ('status', models.CharField(default='pending', max_length=100)),
                ('doctor', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='app.doctorsmodel')),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='app.patientsmodel')),
            ],
            options={
                'db_table': 'AppointmentsModel',
            },
        ),
        migrations.DeleteModel(
            name='ChatMessage',
        ),
    ]
