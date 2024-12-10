from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# 提供静态 HTML 页面
@app.route('/')
def index():
    return send_from_directory('static', '/111/111.html')

@app.route('/uploads', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['audio']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 保存音频为 'audio.wav'，并覆盖已有文件
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'audio.wav')
    file.save(file_path)

    # 返回音频文件 URL，供前端播放
    file_url = 'uploads/audio.wav'

    return jsonify({'message': 'File uploaded successfully', 'audio_url': file_url})


# 提供已上传的文件（用于返回文件）
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)