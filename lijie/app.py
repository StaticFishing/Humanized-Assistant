from flask import Flask, request, jsonify
from flask_cors import CORS  # 导入 flask-cors
from pydub import AudioSegment
import io

app = Flask(__name__)
CORS(app)  # 启用跨域资源共享


@app.route('/upload', methods=['POST'])
def upload():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file found'}), 400

    audio_file = request.files['audio']

    # 读取音频文件
    audio = AudioSegment.from_file(audio_file)

    # 假设我们进行一个简单的处理，比如将音频转为静音的
    audio = audio - 100  # 减少音量处理

    # 将处理后的音频保存为 bytes
    processed_audio = io.BytesIO()
    audio.export(processed_audio, format="wav")
    processed_audio.seek(0)

    # 返回处理结果（这里简单返回静音后的音频大小）
    return jsonify({'result': f'Processed audio size: {len(processed_audio.getvalue())} bytes'})


if __name__ == '__main__':
    app.run(debug=True)
