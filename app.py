from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import torch
import pandas as pd
from datetime import timedelta
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Configure upload and processed folders
UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = './processed'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to upload form (index page)
@app.route('/')
def upload_form():
    return render_template('index.html')

# Route to handle video upload and processing
@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        
        if file.filename == '':
            return 'No selected file'
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            # Process the video
            output_video_path = os.path.join(app.config['PROCESSED_FOLDER'], 'output_shortened.mp4')
            dataset_path = os.path.join(app.config['PROCESSED_FOLDER'], 'shortened_frames_dataset.csv')
            process_video(video_path, output_video_path, dataset_path)

            # Return the processed video download link
            return render_template('download.html', video_link=output_video_path)

    # If the request is GET, render the upload page
    return render_template('upload.html')


# Video processing function (frame extraction, timestamping, and shortening)
def process_video(input_video_path, output_video_path, dataset_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Couldn't open video.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: FPS value is 0. Manually setting to 30 FPS.")
        fps = 30  # Default to 30 FPS if OpenCV fails to retrieve FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_data = []
    frame_number = 0
    selected_frames = []
    skip_count = 2  # Number of frames to skip after processing a frame with detected movement

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = str(timedelta(seconds=int(frame_number / fps)))

        # Perform object detection
        results = model(frame)

        # Check for human detections (class id 0)
        humans_detected = False
        for *box, conf, cls in results.xyxy[0]:  # xyxy format
            if int(cls) == 0:  # 0 corresponds to 'person'
                x1, y1, x2, y2 = map(int, box)
                humans_detected = True

                # Draw bounding box around detected human
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display timestamp above the bounding box
                cv2.putText(
                    frame,
                    timestamp,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )

        if humans_detected:
            selected_frames.append(frame)  # Store the frame with detected human
            frame_data.append({
                'frame_number': frame_number,
                'timestamp': timestamp
            })

            # Skip the next few frames
            for _ in range(skip_count):
                cap.read()
                frame_number += 1

        frame_number += 1

    # Write selected frames to output video
    for frame in selected_frames:
        out.write(frame)

    # Save the dataset to CSV
    df = pd.DataFrame(frame_data)
    df.to_csv(dataset_path, index=False)

    cap.release()
    out.release()

# Route to download the processed video
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    # Ensure necessary directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    app.run(debug=True)
