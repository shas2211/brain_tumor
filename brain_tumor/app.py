from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import time

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load your CNN model
model = load_model("brain_tumor_cnn.h5")

# Class labels
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Tumor info dictionary
tumor_info_dict = {
    'glioma': "Glioma: A tumor that arises from glial cells in the brain. Can be slow-growing or aggressive.",
    'meningioma': "Meningioma: Usually benign tumor forming on the meninges, the brain’s protective layers.",
    'notumor': "No tumor detected: The scan appears normal without signs of a tumor.",
    'pituitary': "Pituitary tumor: A growth in the pituitary gland which can affect hormone levels."
}

# Function to classify image
def classify_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(np.max(predictions))
        return predicted_class_name, confidence
    except Exception as e:
        return f"Error: {e}", 0

# Disable caching
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename != "":
            # Save uploaded image
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Classify image
            predicted_class, conf = classify_image(filepath)

            # Add timestamp to image URL to prevent caching
            timestamp = int(time.time())
            uploaded_image = f"{filepath}?v={timestamp}"

            # Format confidence
            confidence = f"{round(conf * 100, 2)}%"

            # Get tumor info
            tumor_info = tumor_info_dict.get(predicted_class, "")

            return render_template(
                "index.html",
                predicted_class=predicted_class,
                confidence=confidence,
                uploaded_image=uploaded_image,
                tumor_info=tumor_info
            )

    # GET request: reset everything
    return render_template(
        "index.html",
        predicted_class=None,
        confidence=None,
        uploaded_image=None,
        tumor_info=None
    )

if __name__ == "__main__":
    app.run(debug=True)
