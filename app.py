"""Gradio demo: Pneumonia detection from chest X-rays (DenseNet121 transfer learning).

Run after the notebook has produced artifacts/:
    pip install -r requirements.txt
    python app.py
"""
import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow import keras

ART = Path("artifacts")
IMG = json.load(open(ART / "config.json"))["img_size"]   # 224 full run, 160 quick CPU run
CARD = json.load(open(ART / "model_card.json"))
THRESHOLD = CARD["decision_threshold"]
CLASSES = CARD["classes"]

model = keras.models.load_model(ART / "pneumonia_densenet121.keras")
base = next(l for l in model.layers if isinstance(l, keras.Model) and
            any(x.name == "conv5_block16_concat" for x in l.layers))
pre = model.get_layer("preprocess")
head = model.layers[model.layers.index(base) + 1:]
feat_model = keras.Model(base.inputs, [base.get_layer("conv5_block16_concat").output, base.output])


def to_model_input(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    gray = cv2.resize(gray, (IMG, IMG), interpolation=cv2.INTER_AREA)
    return np.stack([gray] * 3, -1)


def gradcam(rgb):
    x = tf.convert_to_tensor(rgb[None].astype("float32"))
    with tf.GradientTape() as tape:
        conv, y = feat_model(pre(x), training=False)
        for layer in head[:-1]:
            y = layer(y, training=False)
        dense = head[-1]  # Grad-CAM on the logit: the sigmoid saturates for confident images -> blank maps
        logit = tf.cast(tf.matmul(tf.cast(y, dense.kernel.dtype), dense.kernel) + dense.bias, tf.float32)[:, 0]
        score = tf.where(logit >= 0, logit, -logit)  # explain the predicted class
    grads = tf.cast(tape.gradient(score, conv), tf.float32)
    conv = tf.cast(conv, tf.float32)
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * tf.reduce_mean(grads, (0, 1, 2)), -1)).numpy()
    cam = cv2.resize(cam / (cam.max() + 1e-8), (IMG, IMG))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[..., ::-1]
    return np.uint8(rgb * 0.6 + heat * 0.4), float(tf.sigmoid(logit)[0])


def predict(image):
    if image is None:
        return None, None, ""
    rgb = to_model_input(np.asarray(image))
    overlay, prob = gradcam(rgb)
    label = CLASSES[int(prob >= THRESHOLD)]
    note = (f"**Prediction: {label}**  \nP(pneumonia) = {prob:.3f} (decision threshold {THRESHOLD:.3f}, "
            f"tuned for high sensitivity)  \n\n⚠️ Research demo only — not a medical diagnosis.")
    return {"PNEUMONIA": prob, "NORMAL": 1 - prob}, overlay, note


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Chest X-ray (JPEG/PNG)"),
    outputs=[gr.Label(num_top_classes=2, label="Probabilities"),
             gr.Image(label="Grad-CAM (where the model looked)"),
             gr.Markdown()],
    title="🩻 Pneumonia Detection — DenseNet121 Transfer Learning",
    description="Upload a pediatric chest X-ray. The model returns a pneumonia probability and a Grad-CAM heatmap.",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
