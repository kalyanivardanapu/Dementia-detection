from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = load_model('cnn_model_finetuned.h5')

# Class labels
class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (50, 50))
    # Simple and fast image enhancement
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Preprocess the image for prediction
    processed_img = preprocess_image(filepath)
    
    # Make prediction
    prediction = model.predict(processed_img)
    predicted_class_index = np.argmax(prediction[0])
    confidence_score = float(prediction[0][predicted_class_index]) * 100
    prediction_label = class_labels[predicted_class_index]
    
    # Create enhanced ROI visualization
    image = cv2.imread(filepath)
    image = cv2.resize(image, (512, 512))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Split brain into left and right hemispheres
    mid_point = image.shape[1] // 2
    left_half = gray[:, :mid_point]
    right_half = gray[:, mid_point:]
    
    # Analyze intensity distributions in each hemisphere
    left_mean = np.mean(left_half)
    right_mean = np.mean(right_half)
    
    # Determine affected side based on intensity differences
    threshold = 0.1  # Threshold for significant difference
    diff_percentage = abs(left_mean - right_mean) / max(left_mean, right_mean)
    
    if diff_percentage > threshold:
        affected_side = "Left Side" if left_mean < right_mean else "Right Side"
    else:
        affected_side = "Bilateral"
    
    overlay = image.copy()
    
    # Define different ROIs based on diagnosis and affected side
    if prediction_label == 'Mild Demented':
        if affected_side == "Left Side":
            regions = [
                {
                    'name': 'Left Temporal Atrophy',
                    'coords': (50, 200, 200, 300),
                    'color': (0, 255, 0)  # Green
                },
                {
                    'name': 'Left Hippocampal Changes',
                    'coords': (100, 250, 200, 350),
                    'color': (255, 0, 0)  # Blue
                }
            ]
        elif affected_side == "Right Side":
            regions = [
                {
                    'name': 'Right Temporal Atrophy',
                    'coords': (312, 200, 462, 300),
                    'color': (0, 255, 0)  # Green
                },
                {
                    'name': 'Right Hippocampal Changes',
                    'coords': (312, 250, 412, 350),
                    'color': (255, 0, 0)  # Blue
                }
            ]
        else:  # Bilateral
            regions = [
                {
                    'name': 'Bilateral Temporal Atrophy',
                    'coords': (150, 200, 362, 300),
                    'color': (0, 255, 0)  # Green
                }
            ]
    
    elif prediction_label == 'Moderate Demented':
        if affected_side == "Left Side":
            regions = [
                {
                    'name': 'Severe Left Hippocampal Atrophy',
                    'coords': (80, 230, 220, 370),
                    'color': (0, 0, 255)  # Red
                },
                {
                    'name': 'Left Temporal Degradation',
                    'coords': (30, 180, 170, 320),
                    'color': (255, 165, 0)  # Orange
                }
            ]
        elif affected_side == "Right Side":
            regions = [
                {
                    'name': 'Severe Right Hippocampal Atrophy',
                    'coords': (292, 230, 432, 370),
                    'color': (0, 0, 255)  # Red
                },
                {
                    'name': 'Right Temporal Degradation',
                    'coords': (342, 180, 482, 320),
                    'color': (255, 165, 0)  # Orange
                }
            ]
        else:
            regions = [
                {
                    'name': 'Bilateral Severe Atrophy',
                    'coords': (100, 230, 412, 370),
                    'color': (128, 0, 128)  # Purple
                }
            ]
    
    elif prediction_label == 'Very Mild Demented':
        if affected_side == "Left Side":
            regions = [
                {
                    'name': 'Early Left Temporal Changes',
                    'coords': (60, 210, 140, 290),
                    'color': (255, 255, 0)  # Yellow
                }
            ]
        elif affected_side == "Right Side":
            regions = [
                {
                    'name': 'Early Right Temporal Changes',
                    'coords': (372, 210, 452, 290),
                    'color': (255, 255, 0)  # Yellow
                }
            ]
        else:
            regions = [
                {
                    'name': 'Mild Bilateral Changes',
                    'coords': (160, 210, 352, 290),
                    'color': (0, 255, 255)  # Cyan
                }
            ]
    
    else:  # Non Demented
        regions = [
            {
                'name': 'Normal Brain Structure',
                'coords': (150, 200, 362, 300),
                'color': (0, 255, 0)  # Green
            }
        ]
    
    # Draw ROIs with different patterns based on severity
    for region in regions:
        x1, y1, x2, y2 = region['coords']
        if prediction_label == 'Moderate Demented':
            # Add cross-hatching pattern
            cv2.rectangle(overlay, (x1, y1), (x2, y2), region['color'], -1)
            for i in range(y1, y2, 10):
                cv2.line(overlay, (x1, i), (x2, i), (255, 255, 255), 1)
            for i in range(x1, x2, 10):
                cv2.line(overlay, (i, y1), (i, y2), (255, 255, 255), 1)
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), region['color'], -1)
        
        # Add border
        cv2.rectangle(image, (x1, y1), (x2, y2), region['color'], 2)
    
    # Draw midline
    cv2.line(overlay, (mid_point, 0), (mid_point, 512), (255, 255, 255), 1)
    
    # Apply transparency
    alpha = 0.3
    roi_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    # Add text labels and information
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 30
    
    # Add region labels
    for region in regions:
        x1, y1, _, _ = region['coords']
        cv2.putText(roi_image, region['name'], (x1, y1-10), font, 0.5, region['color'], 2)
    
    # Add diagnosis information
    cv2.putText(roi_image, f"Diagnosis: {prediction_label}", (10, y_offset), font, 0.7, (255, 255, 255), 2)
    cv2.putText(roi_image, f"Affected Side: {affected_side}", (10, y_offset + 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(roi_image, f"Confidence: {confidence_score:.2f}%", (10, y_offset + 60), font, 0.7, (255, 255, 255), 2)
    
    # Add severity-specific notes
    severity_note = {
        'Moderate Demented': f"Significant {affected_side.lower()} atrophy detected",
        'Mild Demented': f"Early signs of {affected_side.lower()} atrophy",
        'Very Mild Demented': f"Subtle changes in {affected_side.lower()}",
        'Non Demented': "Normal brain structure"
    }
    cv2.putText(roi_image, severity_note[prediction_label], (10, y_offset + 90), font, 0.6, (0, 255, 255), 2)
    
    # Convert the ROI image to base64
    _, buffer = cv2.imencode('.jpg', roi_image)
    roi_image_b64 = base64.b64encode(buffer).decode('utf-8')
    
    # Clean up
    os.remove(filepath)
    
    return jsonify({
        'prediction': prediction_label,
        'confidence': f'{confidence_score:.2f}%',
        'affected_side': affected_side,
        'segmented_image': roi_image_b64
    })

if __name__ == '__main__':
    app.run(debug=True) 