from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
from utils import preprocess_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load pre-trained model
model = load_model('model/fruit_vegetable_classifier.h5')

# Class labels
class_labels = ['Fresh', 'Rotten']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        img = preprocess_image(filepath)

        # Prediction
        preds = model.predict(img)
        pred_label = class_labels[np.argmax(preds)]
        confidence = round(np.max(preds) * 100, 2)

        return render_template('result.html',
                               label=pred_label,
                               confidence=confidence,
                               image_file=filepath)

    return redirect(url_for('index'))


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)