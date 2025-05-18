# Image Classification Web Application

This is a web application that allows users to upload images and get predictions using a TensorFlow Lite model.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure your model file `final_densenet_finetuned_model.tflite` is in the root directory.

2. Start the Flask application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

## Features

- Drag and drop image upload
- Image preview
- Real-time predictions
- Support for PNG, JPG, and JPEG formats
- Modern and responsive UI

## Notes

- Maximum file size: 16MB
- Supported image formats: PNG, JPG, JPEG
- Images are automatically resized to 224x224 pixels before processing
- Uploaded images are temporarily stored and then deleted after processing 