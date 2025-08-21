import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def predict_gender(img_path):
    model = load_model('model/gender_model.h5')
    img = image.load_img(img_path, target_size=(150,150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    label = 'Male' if pred > 0.5 else 'Female'
    print(f"Prediction: {label} (Confidence: {pred:.2f})")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        predict_gender(sys.argv[1])
# Just testing perpose
