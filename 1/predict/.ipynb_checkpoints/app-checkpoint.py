from flask import Flask, render_template, request
import numpy as np
import requests
from PIL import Image
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST' and 'image' in request.files:
        image_file = request.files['image']
        image_data = image_file.read()
        img = Image.open(BytesIO(image_data))
        img = img.convert('RGB')
        img = img.resize((32, 32))  # Resize image to 32x32
        img_array = np.array(img) / 255.0  # Normalize pixel values

        if img_array.shape != (32, 32, 3):
            return "Error: Invalid image dimensions. Please upload an image with dimensions (32, 32, 3)."

        img_array = np.expand_dims(img_array, axis=0)

        data = {"instances": img_array.tolist()}

        response = requests.post('http://localhost:8501/v1/models/my_model/versions/1:predict', json=data)

        try:
            print("Response JSON:", response.json())
            predictions = response.json()['predictions'][0]
            predicted_class = np.argmax(predictions)
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            predicted_category = class_names[predicted_class]
            return render_template('prediction.html', prediction=predicted_category)
        except Exception as e:
            print("Error processing predictions:", e)
            return "Error processing predictions"
    else:
        return "Error: No image uploaded or invalid request method"

if __name__ == '__main__':
    app.run(debug=True)