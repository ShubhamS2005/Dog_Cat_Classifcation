from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img,img_to_array 
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('cat_dog_custom2.h5')

# Image dimensions
img_width, img_height = 128, 128

def predict_image(image_path):
    img = load_img(image_path, target_size=(img_width, img_height))
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    return "Dog" if prediction[0] > 0.5 else "Cat"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    
    if not os.path.isdir("static/uploads"):
        os.mkdir("static/uploads")

    for filename in os.listdir("static/uploads"):
        file_path=os.path.join("static/uploads",filename)
        if(file_path!=os.path.join("static/uploads",file.filename)):
            os.remove(file_path)

    filepath = f"static/uploads/{file.filename}"
    file.save(filepath)

    result = predict_image(filepath)
    
    return render_template('index.html', prediction=result, filepath=filepath)

if __name__ == '__main__':
    app.run(debug=True)
