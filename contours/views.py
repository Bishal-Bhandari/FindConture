import base64
import cv2
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rembg import remove


@csrf_exempt
def detect_contours(request):
    if request.method == 'POST':
        try:
            image_file = request.FILES['image']  # Get the image from the request
            img_fs = image_file.read()  # Read the image file into bytes
            np_ary = np.frombuffer(img_fs, np.uint8)  # Convert to numpy array

            img = cv2.imdecode(np_ary, cv2.IMREAD_COLOR)  # Decode the image

            bnw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            gus_blr = cv2.GaussianBlur(bnw, (5, 5), 0)  # Apply Gaussian blur
            _, roi = cv2.threshold(gus_blr, 0, 255, cv2.THRESH_BINARY)  # Thresholding

            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours

            # Prepare contour data
            contour_dict = {i + 1: contour.reshape(-1, 2).tolist() for i, contour in enumerate(contours)}

            # Draw contours on the image
            cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

            # Convert the processed image to base64 for rendering in HTML
            _, buffer = cv2.imencode('.png', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            return JsonResponse({'contours': contour_dict, 'image_data': img_base64}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


def upload_page(request):
    return render(request, 'contours/index.html')  # Render the HTML page
