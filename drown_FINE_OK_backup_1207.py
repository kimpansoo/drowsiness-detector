import sys
import winsound
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QLabel, QRadioButton, QButtonGroup, QSlider, QWidget,
                             QTextEdit, QPushButton)
from PyQt5.QtCore import Qt, QTimer, QTime
from PyQt5.QtGui import QImage, QPixmap


class DrowsinessDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setup_face_detection()
        self.initialize_camera()  # 카메라 초기화 메서드 분리

    def initialize_camera(self):
        """카메라를 안전하게 초기화하는 메서드"""
        # 여러 카메라 인덱스 시도
        camera_indices = [1,0,2]
        for index in camera_indices:
            self.cap = cv2.VideoCapture(index)
            if self.cap.isOpened():
                print(f"Camera {index} successfully opened")
                return
        raise RuntimeError("No camera could be initialized")

    def initUI(self):
        self.setWindowTitle('졸음감지기 (Drowsiness Detector)')
        self.setGeometry(500, 100, 500, 380)  # 리셋 버튼을 위해 약간 높이 조정

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()

        # Title
        title_label = QLabel('졸음감지기 (Drowsiness Detector)')
        title_label.setStyleSheet("font-size: 38px; font-weight: bold; text-align: center; margin: 20px;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Video and Control Panel Layout
        video_control_layout = QHBoxLayout()

        # Video Display
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)  # 영상 크기 고정
        self.video_label.setStyleSheet("border: 2px solid black; background-color: #000;")
        video_control_layout.addWidget(self.video_label)

        # Control Panel
        control_layout = QVBoxLayout()

        # EAR Sensitivity Settings
        sensitivity_label = QLabel('       민감도 설정')
        sensitivity_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 20px;")
        control_layout.addWidget(sensitivity_label)

        sensitivity_group = QVBoxLayout()
        self.radio_group = QButtonGroup(self)
        self.sensitivity = 0.16  # Default sensitivity

        for i, value in enumerate(np.linspace(0.14, 0.22, 9)):
            radio_button = QRadioButton(f"{i + 1} 단계")
            radio_button.value = value
            self.radio_group.addButton(radio_button, i)
            if i % 3 == 0:
                row_layout = QHBoxLayout()
                sensitivity_group.addLayout(row_layout)
            row_layout.addWidget(radio_button)

        self.radio_group.buttonClicked.connect(self.update_sensitivity)
        control_layout.addLayout(sensitivity_group)

        # Warning Sound Volume Dial
        volume_label = QLabel('     경고음 주파수 조절')
        volume_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 20px;")
        control_layout.addWidget(volume_label)

        self.volume_dial = QSlider(Qt.Horizontal)
        self.volume_dial.setRange(50, 3000)
        self.volume_dial.setValue(2000)  # Default frequency
        control_layout.addWidget(self.volume_dial)

        # 리셋 버튼 추가
        reset_button = QPushButton('시스템 리셋')
        reset_button.setStyleSheet("""
            QPushButton {
                background-color: #ff6b6b; 
                color: white; 
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #ff5252;
            }
        """)
        reset_button.clicked.connect(self.reset_system)
        control_layout.addWidget(reset_button)

        # Log Area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFixedWidth(250)  # 감지 로그 창 폭 설정
        self.log_area.setFixedHeight(230)
        self.log_area.setStyleSheet("background-color: #f5f5f5; border: 1px solid gray;")
        control_layout.addWidget(QLabel('검지 로그:'))
        control_layout.addWidget(self.log_area)

        video_control_layout.addLayout(control_layout)
        main_layout.addLayout(video_control_layout)
        central_widget.setLayout(main_layout)

        # Video Capture Timer Setup
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30 ms intervals

        self.drowsy_count = 0
        self.max_drowsy_count = 10

    def setup_face_detection(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def reset_system(self):
        """전체 시스템 리셋 메서드"""
        try:
            # 카메라 재초기화
            if hasattr(self, 'cap'):
                self.cap.release()
            self.initialize_camera()

            # 졸음 카운트 초기화
            self.drowsy_count = 0

            # UI 요소 초기화
            self.volume_dial.setValue(2000)  # 기본 볼륨으로

            # 라디오 버튼 첫 번째(기본) 항목으로 리셋
            first_radio_button = self.radio_group.button(0)
            if first_radio_button:
                first_radio_button.setChecked(True)
                self.sensitivity = 0.16

            # 로그 영역 초기화
            self.log_area.clear()
            current_time = QTime.currentTime().toString('HH:mm:ss')
            self.log_area.append(f"{current_time} - 시스템 완전 초기화 완료")

        except Exception as e:
            self.log_area.append(f"리셋 중 오류 발생: {str(e)}")

    def update_sensitivity(self, button):
        self.sensitivity = button.value
        self.log_area.append(f"{QTime.currentTime().toString('HH:mm:ss')} -       민감도 설정: {self.sensitivity:.2f}")

    def calculate_ear(self, landmarks, eye_indices):
        def distance(p1, p2):
            return np.linalg.norm(np.array(p1) - np.array(p2))

        vertical1 = distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
        vertical2 = distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
        horizontal = distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])

        return (vertical1 + vertical2) / (2 * horizontal)

    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                # 프레임 읽기 실패 시 카메라 재초기화
                self.log_area.append("카메라 프레임 읽기 실패. 재연결 시도...")
                self.initialize_camera()
                return

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
                self.log_area.append(f"{QTime.currentTime().toString('HH:mm:ss')} - 졸음 감지됨!!!!!!")
                winsound.Beep(self.volume_dial.value(), 300)

            # 프레임 변환 및 표시
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))

        except Exception as e:
            self.log_area.append(f"프레임 처리 중 오류: {str(e)}")

    def closeEvent(self, event):
        self.cap.release()
        self.face_mesh.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    detector = DrowsinessDetector()
    detector.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()