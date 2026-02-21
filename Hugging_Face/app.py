import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("best_cnn_RGB.keras")

# ⚠️ IMPORTANT:
# This label order MUST match train_generator.class_indices
labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']

def predict_emotion(image):
    if image is None:
        return None, None, "No image uploaded"

    # Keep original image
    original_image = image.copy()

    # Preprocess (same as training)
    img = cv2.resize(image, (48, 48))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)[0]
    top_idx = np.argmax(preds)
    emotion = labels[top_idx]
    confidence = preds[top_idx] * 100

    # Overlay prediction
    predicted_image = original_image.copy()
    cv2.putText(
        predicted_image,
        f"Predicted: {emotion} ({confidence:.2f}%)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # Return:
    # 1️⃣ Original image
    # 2️⃣ Image with prediction
    # 3️⃣ Top-3 probabilities
    return (
        original_image,
        predicted_image,
        {labels[i]: float(preds[i]) for i in range(len(labels))}
    )

# Gradio UI
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="numpy", label="Upload Face Image"),
    outputs=[
        gr.Image(label="Actual (Uploaded) Image"),
        gr.Image(label="Predicted Image"),
        gr.Label(num_top_classes=3, label="Prediction Probabilities")
    ],
    title="Facial Emotion Recognition (RGB)",
    description="Upload a face image to view predicted emotion and confidence."
)

if __name__ == "__main__":
    interface.launch()