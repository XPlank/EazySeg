🖼 EazySeg Web App

This website allows you to upload one or multiple images and automatically performs real-time object detection using a YOLOv11 deep learning model. Finally drawing Bounding boxes and segregating them weather Dry or Wet waste.

✨ Key Features

📤 Multi-image upload – Upload several images at once for batch processing.

🤖 YOLO-powered detection – Uses your trained best.pt model to detect objects accurately.

🖍 Bounding boxes & labels – Detected objects are highlighted with bounding boxes and labels directly on the images.

📂 Results gallery – New uploads appear at the top, while previous detections remain accessible below.

⬇ Download processed images – Save the detection results with bounding boxes to your local device.

🎨 Modern UI – Clean, responsive Bootstrap-based design for a smooth user experience.


🚀 How It Works

1. Upload one or more images using the file selector.


2. The backend Flask server runs inference using the YOLOv11 model (best.pt).


3. The app overlays bounding boxes and labels on detected objects.


4. Results are displayed in the gallery, with options to view or download.



🛠 Technology Stack

Frontend: HTML5, CSS, Bootstrap 5, JavaScript

Backend: Python (Flask)

Deep Learning: YOLOv11 (ultralytics) + PyTorch

Utilities: NumPy, OpenCV
