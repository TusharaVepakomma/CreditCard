
from django.conf.urls import url
from django.contrib import admin
from django import views

from App.views import index

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^card/', index),

]
