import sys
import sounddevice as sd
import wave
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import QSize


class AudioRecorder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("录音器")
        self.setGeometry(100, 100, 300, 200)

        # UI 元素
        self.layout = QVBoxLayout()
        self.status_label = QLabel("按住按钮开始录音，松开结束录音")
        self.layout.addWidget(self.status_label)

        self.record_button = QPushButton("")
        self.record_button.setCheckable(True)
        self.record_button.pressed.connect(self.start_recording)
        self.record_button.released.connect(self.stop_recording)

        # 自定义按钮样式
        self.record_button.setFixedSize(100, 100)  # 设置按钮大小
        self.record_button.setStyleSheet("""
            QPushButton {
                border: 2px solid #FF0000;
                border-radius: 50px; /* 半径 = 宽高的一半，按钮变为圆形 */
                background-color: #FF4D4D;
            }
            QPushButton:pressed {
                background-color: #FF0000;
            }
        """)
        self.layout.addWidget(self.record_button)  # 居中对齐

        # 主窗口设置
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # 音频录制相关
        self.recording = False
        self.frames = []
        self.stream = None
        self.fs = 44100  # 采样率
        self.channels = 2  # 声道

    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.frames = []
            self.stream = sd.InputStream(
                samplerate=self.fs, channels=self.channels, callback=self.audio_callback
            )
            self.stream.start()
            self.status_label.setText("录音中...")

    def audio_callback(self, indata, frames, time, status):
        """实时获取音频数据的回调函数"""
        if status:
            print(status)
        self.frames.append(indata.copy())

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.stream.stop()
            self.stream.close()

            # 保存音频到文件
            file_path = "output.wav"
            self.save_audio(file_path)

            self.status_label.setText(f"录音完成，保存到 {file_path}")

    def save_audio(self, file_path):
        """保存录制的音频到文件"""
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.fs)
            wf.writeframes(b''.join(self.frames))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioRecorder()
    window.show()
    sys.exit(app.exec_())
