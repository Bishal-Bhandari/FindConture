from django.urls import path
from .views import detect_contours, upload_page

urlpatterns = [
    path('', upload_page, name='upload_page'),  # For rendering the HTML page
    path('detect_contours/', detect_contours, name='detect_contours'),  # For detecting contours
]
