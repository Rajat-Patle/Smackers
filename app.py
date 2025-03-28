import os
from flask import Flask, render_template, request
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Ensure the 'static/uploads' folder exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load YOLO model
model = YOLO('models/yolov8n.pt')  # Ensure the correct model path

def calculate_percentage(count, max_count):
    return (count / max_count * 100) if max_count > 0 else 0

@app.route("/", methods=["GET", "POST"])
def home():
    # Default values for first-time load
    signal_times = [30, 30, 30, 30]  
    vehicles_counts = [0, 0, 0, 0]  
    signal_status = ["Red", "Red", "Red", "Red"]  

    if request.method == "POST":
        uploaded_files = []
        
        # Process all four uploaded images
        for i in range(4):
            file = request.files.get(f"signal{i+1}")
            if file and file.filename != "":
                image_path = os.path.join(app.config["UPLOAD_FOLDER"], f"signal{i+1}.jpg")
                file.save(image_path)
                uploaded_files.append(image_path)

                # Load image and detect vehicles
                img = cv2.imread(image_path)
                if img is not None:
                    results = model(img)
                    result = results[0]
                    vehicles_counts[i] = len(result.boxes.xywh)

        if any(uploaded_files):  # Only process signals if images were uploaded
            max_vehicles = max(vehicles_counts)
            max_index = vehicles_counts.index(max_vehicles)  # Most crowded signal

            # Assign signal times dynamically
            for i in range(4):
                if i == max_index:
                    signal_times[i] = 60  # More time for more vehicles
                    signal_status[i] = "Green"
                else:
                    signal_times[i] = 30  # Default signal time
                    signal_status[i] = "Red"

            # Set the next signal to Yellow
            next_signal = (max_index + 1) % 4
            signal_status[next_signal] = "Yellow"

    # Prevent Zero Division and Calculate Percentages
    max_vehicle_count = max(vehicles_counts) if any(vehicles_counts) else 1
    percentages = [calculate_percentage(count, max_vehicle_count) for count in vehicles_counts]

    return render_template(
        "index.html",
        signal_times=signal_times,
        vehicles_counts=vehicles_counts,
        signal_status=signal_status,
        percentages=percentages,
        upload_folder=UPLOAD_FOLDER
    )

if __name__ == "__main__":
    app.run(debug=True)
