import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf # Using full TensorFlow

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'final_densenet_finetuned_model.tflite'
IMAGE_SIZE = 224 # Assuming your model expects 224x224 images

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Define Class Names and English Translations ---
CLASS_NAMES_IT = [
    "cane", "cavallo", "elefante", "farfalla", "gallina", 
    "gatto", "mucca", "pecora", "ragno", "scoiattolo"
]

CLASS_NAMES_EN = {
    "cane": "Dog",
    "cavallo": "Horse",
    "elefante": "Elephant",
    "farfalla": "Butterfly",
    "gallina": "Chicken",
    "gatto": "Cat",
    "mucca": "Cow",
    "pecora": "Sheep",
    "ragno": "Spider",
    "scoiattolo": "Squirrel"
}

# --- Load the TFLite model using TensorFlow ---
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Successfully loaded TFLite model from {MODEL_PATH}")
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")

    # CLASS_NAMES definition moved up with translations

except Exception as e:
    print(f"Error loading TFLite model: {e}")
    interpreter = None # Ensure interpreter is None if loading failed

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Ensure RGB
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:
        return jsonify({'error': 'Model not loaded or failed to load. Check server logs.'}), 500

    print(f"Request files: {list(request.files.keys())}") # DEBUG: What file parts are received?

    if 'file' not in request.files:
        print("DEBUG: 'file' not in request.files") # DEBUG
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    print(f"DEBUG: Received file object: {file}") # DEBUG
    print(f"DEBUG: Filename from browser: {file.filename}") # DEBUG

    if file.filename == '':
        print("DEBUG: file.filename is empty") # DEBUG
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            
            processed_image = preprocess_image(filepath)
            
            # Ensure input tensor is the correct type (usually float32 for TFLite)
            if input_details[0]['dtype'] == np.float32 and processed_image.dtype != np.float32:
                processed_image = processed_image.astype(np.float32)
            elif input_details[0]['dtype'] == np.uint8 and processed_image.dtype != np.uint8:
                 # If model expects uint8 (0-255), adjust preprocessing (no /255.0) and cast
                processed_image = (processed_image * 255).astype(np.uint8) 

            interpreter.set_tensor(input_details[0]['index'], processed_image)
            interpreter.invoke()
            prediction_output = interpreter.get_tensor(output_details[0]['index'])
            
            # Process the prediction output for multi-class classification
            # prediction_output is likely a 2D array, e.g., [[score1, score2, ...]]
            if prediction_output.ndim == 2 and prediction_output.shape[0] == 1:
                scores = prediction_output[0] # Get the actual array of scores
                predicted_class_index = np.argmax(scores)
                
                # Get Italian class name
                predicted_class_it = CLASS_NAMES_IT[predicted_class_index]
                # Get English translation
                predicted_class_en = CLASS_NAMES_EN.get(predicted_class_it, "Unknown") # Fallback
                
                confidence_score = float(scores[predicted_class_index])
                
                result_value = f"{predicted_class_it.capitalize()} / {predicted_class_en} (Confidence: {confidence_score:.2f})"
            else:
                # Fallback or error if output shape is not as expected
                print(f"Unexpected prediction output shape: {prediction_output.shape}")
                result_value = "Error processing prediction output (unexpected shape)"
                return jsonify({'error': result_value, 'success': False}), 500

            return jsonify({
                'success': True,
                'prediction': result_value 
            })
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath) # Clean up uploaded file
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True) 