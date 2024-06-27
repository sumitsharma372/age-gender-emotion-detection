import cv2
import numpy as np
from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .inference import inference
from django.core.files.storage import default_storage

def index(request):
    return render(request, 'detection/index.html')

def gen_frames():  # Generator function for live webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result_frame = inference(frame)
        ret, buffer = cv2.imencode('.jpg', result_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        file_path = default_storage.save('uploaded_images/' + image.name, image)
        image = cv2.imread(file_path)
        result_image = inference(image)
        _, img_encoded = cv2.imencode('.jpg', result_image)
        response = HttpResponse(img_encoded.tobytes(), content_type='image/jpeg')
        return response
    return render(request, 'detection/upload_image.html')


@csrf_exempt
def upload_video(request):
    if request.method == 'POST' and request.FILES['video']:
        video = request.FILES['video']
        file_path = default_storage.save('uploaded_videos/' + video.name, video)
        cap = cv2.VideoCapture(file_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result_frame = inference(frame)
            frames.append(result_frame)
        cap.release()
        result_video_path = 'processed_videos/' + video.name.split('.')[0] + '_processed.avi'
        out = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc(*'DIVX'), 20.0, (frames[0].shape[1], frames[0].shape[0]))
        for frame in frames:
            out.write(frame)
        out.release()
        with open(result_video_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='video/avi')
            response['Content-Disposition'] = f'attachment; filename="{result_video_path.split("/")[-1]}"'
            return response
    return render(request, 'detection/upload_video.html')





