# Generated by Django 4.2.1 on 2023-06-05 08:51

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0003_userquery_privious_upload'),
    ]

    operations = [
        migrations.CreateModel(
            name='Files',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(upload_to='files', verbose_name='file')),
                ('email', models.EmailField(max_length=254)),
            ],
        ),
        migrations.RemoveField(
            model_name='userquery',
            name='privious_upload',
        ),
        migrations.AlterField(
            model_name='userquery',
            name='file',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='users.files', verbose_name='files'),
        ),
    ]
