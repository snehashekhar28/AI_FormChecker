import time
from flask import Flask, request, Response, jsonify, url_for, send_from_directory
from flask_socketio import SocketIO, emit
import asyncio
import threading
import base64
import wave
import os, uuid, platform, subprocess
from werkzeug.utils import secure_filename

from pose_estimation import get_video_data, generate_natural_language_feedback
from classify_squat import classify

Q_POLLING_RATE = 0.1

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow cross-origin requests

# BASE_DIR is the directory where app.py actually lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Then define both folders relative to BASE_DIR
UPLOAD_FOLDER    = os.path.join(BASE_DIR, 'uploads')
ANNOTATED_FOLDER = os.path.join(BASE_DIR, 'processed')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)


# Classify the squat and generate natural feedback asyncronously
def classify_and_generate(distance_mat, results_dict, sid):
    print("generating results")
    try:
        prob = round(classify(distance_mat) * 10, 1)
        feedback = generate_natural_language_feedback(results_dict)

        print("sending results:", prob, feedback)
        socketio.emit(
            'results',
            {
                'score': prob,
                'feedback': feedback
            },
            room=sid
        )
    except Exception as e:
        app.logger.exception("classify_and_generate failed")


@app.route('/')
def hello_world():
    return 'Hello World'

@app.route('/upload', methods=['POST'])
def upload_video():
    print("receieved upload request")
    # 1. Grab the FileStorage object
    if 'video' not in request.files:
        return {'error': 'No file part "video"'}, 400
    video_file = request.files['video']

    # 2. Sanitize & choose a filename
    filename = secure_filename(video_file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)

    # 3. Save to disk
    video_file.save(save_path)
    print("Saved upload to", save_path)

    # 4. Process with Mediapipe…
    annotated = os.path.join(ANNOTATED_FOLDER, f"mp_{unique_name}")
    frame_pose_matrix, results_dict = get_video_data(video_path=save_path, save_vid=True, save_path=annotated, display_vid=False)

    # grab the Socket.IO session ID from a header
    sid = request.headers.get('X-Socket-ID')

    # start classification and feedback generation process
    threading.Thread(
        target=classify_and_generate,
        args=(frame_pose_matrix, results_dict, sid),
        daemon=True
    ).start()

    # return the annotated video URL immediately
    fetch_url = url_for('serve_processed', filename=os.path.basename(annotated), _external=True)
    return {'processedVideoUrl': fetch_url}


@app.route('/processed/<filename>')
def serve_processed(filename):
    res = send_from_directory(ANNOTATED_FOLDER, filename, mimetype='video/mp4', as_attachment=False)
    return res

# Handle connections
@socketio.on("connect")
def connect():
    # The request.sid is a unique ID for the client connection.
    # It is added by SocketIO
    print(f'Client connected: {request.sid}')

# Handle disconnections
@socketio.on("disconnect")
def disconnect():
    print(f'Client disconnected: {request.sid}')
    os.remove(ANNOTATED_FOLDER)
    os.remove(UPLOAD_FOLDER)

# main driver function
if __name__ == '__main__':
    print("starting server")
    socketio.run(app, host='0.0.0.0', port=5000, use_reloader=False)
