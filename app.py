from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
import os
import uuid
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv11 model
model = YOLO('model/best.pt')

def get_existing_files():
    """Get list of existing files in upload folder, sorted by creation time"""
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        if files:
            return sorted(files,
                         key=lambda x: os.path.getctime(os.path.join(app.config['UPLOAD_FOLDER'], x)),
                         reverse=True)
        return []
    except:
        return []

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    filenames = get_existing_files()
    
    if request.method == 'POST':
        # Check if files were uploaded
        if 'images' not in request.files:
            return render_template('index.html', uploaded=bool(filenames), filenames=filenames, error="No files selected")
        
        files = request.files.getlist('images')
        
        # Filter out empty files
        valid_files = [f for f in files if f.filename != '' and allowed_file(f.filename)]
        
        if not valid_files:
            return render_template('index.html', uploaded=bool(filenames), filenames=filenames, error="No valid image files selected")
        
        new_filenames = []
        processed_count = 0
        detection_results = []
        total_detections = 0
        class_counts = {}
        
        for file in valid_files:
            try:
                # Get file extension from original filename
                original_ext = file.filename.rsplit('.', 1)[1].lower()
                filename = f"{str(uuid.uuid4())}.{original_ext}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Save the uploaded file
                file.save(filepath)
                
                # Run YOLOv11 inference
                results = model(filepath)
                
                # Extract detection information
                detections = []
                if results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        conf = float(boxes.conf[i]) if boxes.conf is not None else 0.0
                        cls = int(boxes.cls[i]) if boxes.cls is not None else 0
                        class_name = results[0].names[cls] if results[0].names else f"Class_{cls}"
                        
                        detections.append({
                            'class': class_name,
                            'confidence': conf * 100,  # Convert to percentage
                            'bbox': boxes.xywh[i].tolist() if boxes.xywh is not None else []
                        })
                        
                        # Update class counts
                        if class_name in class_counts:
                            class_counts[class_name] += 1
                        else:
                            class_counts[class_name] = 1
                        
                        total_detections += 1
                
                # Store detection results
                detection_results.append({
                    'filename': filename,
                    'detections': detections,
                    'detection_count': len(detections)
                })
                
                # Save the result image (overwrite the original with detection results)
                results[0].save(filename=filepath)
                
                new_filenames.append(filename)
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing file {file.filename}: {str(e)}")
                continue
        
        # Update filenames list with new files at the beginning
        filenames = new_filenames + [f for f in filenames if f not in new_filenames]
        
        # Create summary statistics
        summary_stats = {
            'total_images': processed_count,
            'total_detections': total_detections,
            'class_counts': class_counts,
            'avg_detections': round(total_detections / processed_count, 1) if processed_count > 0 else 0
        }
        
        success_message = f"Successfully processed {processed_count} image(s) with {total_detections} detections" if processed_count > 0 else None
        error_message = None if processed_count > 0 else "Failed to process any images"
        
        return render_template('index.html', 
                             uploaded=bool(filenames), 
                             filenames=filenames, 
                             detection_results=detection_results,
                             summary_stats=summary_stats,
                             success=success_message,
                             error=error_message)
    
    return render_template('index.html', uploaded=bool(filenames), filenames=filenames)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
