import cv2
import numpy as np
import mediapipe as mp
import base64
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

class DrowsinessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.sensitivity = 0.16
        self.drowsy_count = 0
        self.max_drowsy_count = 10

    def calculate_ear(self, landmarks, eye_indices):
        def distance(p1, p2):
            return np.linalg.norm(np.array(p1) - np.array(p2))

        vertical1 = distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
        vertical2 = distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
        horizontal = distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])

        return (vertical1 + vertical2) / (2 * horizontal)

    def detect_drowsiness(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        is_drowsy = False
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                             for lm in face_landmarks.landmark]

                left_eye = self.calculate_ear(landmarks, [33, 160, 158, 133, 153, 144])
                right_eye = self.calculate_ear(landmarks, [362, 385, 387, 263, 373, 380])
                ear = (left_eye + right_eye) / 2.0

                if ear < self.sensitivity:
                    self.drowsy_count += 1
                    is_drowsy = self.drowsy_count > self.max_drowsy_count
                else:
                    self.drowsy_count = max(0, self.drowsy_count - 1)

        if is_drowsy:
            cv2.putText(frame, "DROWSY!!!", (60, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

        return frame, is_drowsy

detector = DrowsinessDetector()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def gen_frames():
    cap = cv2.VideoCapture(0)  # 웹캠 인덱스
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame, is_drowsy = detector.detect_drowsiness(frame)
        
        # 프레임을 base64로 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        socketio.emit('video_frame', {'frame': frame_base64, 'is_drowsy': is_drowsy})
    
    cap.release()

@socketio.on('start_stream')
def start_stream():
    socketio.start_background_task(gen_frames)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
