import cv2
import base64
import time
from flask import Flask, render_template
from flask_socketio import SocketIO
from io import BytesIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins (or specify specific domains)
# socketio = SocketIO(app)

def stream_video():
    video_path = 'video.mp4'  # Path to the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
            continue

        # Convert the frame to base64
        _, buffer = cv2.imencode('.jpg', frame)  # Convert frame to JPEG
        img_str = base64.b64encode(buffer).decode('utf-8')

        # Emit the frame to all connected clients
        socketio.emit('video_frame', {'image_data': img_str})

        time.sleep(0.033)  # Delay to match ~30 FPS (adjust based on video FPS)

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.start_background_task(stream_video)
    socketio.run(app, host='0.0.0.0', port=5000)
