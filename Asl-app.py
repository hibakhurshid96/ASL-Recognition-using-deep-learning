import cv2
import numpy as np
from transformers import ViTFeatureExtractor, TFAutoModelForImageClassification
#from tensorflow.keras.models import load_model
import tensorflow as tf

# === Load the model and feature extractor ===
model_path = "visiontransformer"
model = TFAutoModelForImageClassification.from_pretrained(model_path)
#feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)

from transformers import AutoImageProcessor
feature_extractor = AutoImageProcessor.from_pretrained(model_path)


# === Class labels (adjust if needed) ===
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U','V','W','X','Y']

# === Start webcam ===
cap = cv2.VideoCapture(0)

print("Starting ASL Real-Time Prediction. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop/center the hand if needed (you can improve later)
    img = cv2.resize(frame, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess
    inputs = feature_extractor(images=img_rgb, return_tensors="tf")
    logits = model(**inputs).logits
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    prediction = class_labels[predicted_class]

    # Display result
    cv2.putText(frame, f"Prediction: {prediction}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("ASL ViT Predictor", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
