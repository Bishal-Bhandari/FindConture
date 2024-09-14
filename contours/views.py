import base64
import cv2
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .background_detection import detect_background

@csrf_exempt
def detect_contours(request):
    if request.method == 'POST':
        try:
            # Extract the image and number from the request
            image_file = request.FILES.get('image')

            # getting number for threshold value into the cv threshold setting
            number = request.POST.get('number')  # Get the number from the text box
            if number:
                try:
                    threshold_value = int(number)  # Convert the number to an integer
                except ValueError:
                    return JsonResponse({'error': 'Invalid number format'}, status=400)
            else:
                return JsonResponse({'error': 'No number provided'}, status=400)

            # threshold_value = int(detect_background(image_file))

            if image_file:
                img_fs = image_file.read()
                np_ary = np.frombuffer(img_fs, np.uint8)
                img = cv2.imdecode(np_ary, cv2.IMREAD_COLOR)
                print('hello1')
                bnw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gus_blr = cv2.GaussianBlur(bnw, (5, 5), 0)

                # Use the dynamic threshold value
                _, roi = cv2.threshold(gus_blr, threshold_value, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contour_dict = {i + 1: contour.reshape(-1, 2).tolist() for i, contour in enumerate(contours)}

                cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

                _, buffer = cv2.imencode('.png', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')

                return JsonResponse({'contours': contour_dict, 'image_data': img_base64, 'number': threshold_value},
                                    status=200)
            else:
                return JsonResponse({'error': 'No image file provided'}, status=400)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


def upload_page(request):
    return render(request, 'index.html')
