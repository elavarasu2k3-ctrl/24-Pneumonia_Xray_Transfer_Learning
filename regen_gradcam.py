"""Regenerate Grad-CAM figures from the trained model (no retraining) and patch them into the executed notebook.
Usage (inside the pneumonia_tf env, from the project folder):  python regen_gradcam.py
"""
import base64, json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

ART = Path("artifacts")
SRC_NB, EXEC_NB = Path("Pneumonia_Transfer_Learning.ipynb"), Path("Pneumonia_Transfer_Learning_executed.ipynb")
IMG = json.load(open(ART / "config.json"))["img_size"]
THRESH = json.load(open(ART / "model_card.json"))["decision_threshold"]

# Reuse the exact (fixed) Grad-CAM code from the source notebook's Phase 10 cell
src = json.load(open(SRC_NB, encoding="utf-8"))
gc_cell = next(c for c in src["cells"] if c["cell_type"] == "code" and "def gradcam" in "".join(c["source"]))
exec("".join(gc_cell["source"]), globals())

model = keras.models.load_model(ART / "pneumonia_densenet121.keras")
test_df = pd.read_csv(ART / "test_split.csv")
test_df["path"] = test_df.path.str.replace("\\", "/", regex=False)

def load_batch(paths):
    out = []
    for p in paths:
        im = tf.io.decode_jpeg(tf.io.read_file(p), channels=1)
        im = tf.image.grayscale_to_rgb(tf.image.resize(im, (IMG, IMG), antialias=True))
        out.append(im)
    return tf.stack(out)

probs = []
for i in range(0, len(test_df), 32):
    probs.append(model.predict(load_batch(test_df.path.values[i:i + 32]), verbose=0).ravel())
test_df["prob"] = np.concatenate(probs)
test_df["pred"] = (test_df.prob >= THRESH).astype(int)
test_df["error"] = np.where((test_df.y == 1) & (test_df.pred == 0), "FN",
                    np.where((test_df.y == 0) & (test_df.pred == 1), "FP", "correct"))

# Figure 1: correct cases, explained for the predicted class
picks = pd.concat([
    test_df[(test_df.y == 1) & (test_df.pred == 1)].nlargest(3, "prob").assign(case="True Positive"),
    test_df[(test_df.y == 0) & (test_df.pred == 0)].nsmallest(3, "prob").assign(case="True Negative"),
])
fig, axes = plt.subplots(2, 6, figsize=(20, 7))
for i, (_, r) in enumerate(picks.iterrows()):
    img = load_for_view(r.path); cam, s = gradcam(img, model)
    axes[0, i].imshow(img); axes[0, i].set_title(f"{r.case}\n{r.label}", fontsize=9); axes[0, i].axis("off")
    axes[1, i].imshow(overlay(img, cam)); axes[1, i].set_title(f"P(pneumonia)={s:.2f}", fontsize=9); axes[1, i].axis("off")
plt.suptitle("Grad-CAM for the predicted class: where the model looks"); plt.tight_layout()
plt.savefig(ART / "gradcam_examples.png", dpi=120); plt.close()

# Figure 2: hardest mistakes, pneumonia evidence
errs = pd.concat([test_df[test_df.error == "FN"].nsmallest(4, "prob"),
                  test_df[test_df.error == "FP"].nlargest(4, "prob")])
if len(errs):
    fig, axes = plt.subplots(2, len(errs), figsize=(3 * len(errs), 6.5))
    axes = np.array(axes).reshape(2, -1)
    for i, (_, r) in enumerate(errs.iterrows()):
        img = load_for_view(r.path); cam, s = gradcam(img, model, target="pneumonia")
        axes[0, i].imshow(img); axes[0, i].set_title(f"{r.error} | {r.subtype}", fontsize=9); axes[0, i].axis("off")
        axes[1, i].imshow(overlay(img, cam)); axes[1, i].set_title(f"p={s:.2f}", fontsize=9); axes[1, i].axis("off")
    plt.suptitle("Hardest mistakes (top: X-ray, bottom: Grad-CAM of pneumonia evidence)"); plt.tight_layout()
    plt.savefig(ART / "gradcam_errors.png", dpi=120); plt.close()

# Patch executed notebook: new sources for the Grad-CAM cells + new images
ex = json.load(open(EXEC_NB, encoding="utf-8"))
def b64(p): return base64.b64encode(open(p, "rb").read()).decode()
def img_output(p): return {"output_type": "display_data", "metadata": {}, "data": {"image/png": b64(p), "text/plain": ["<Figure>"]}}
src_by_key = {}
for c in src["cells"]:
    t = "".join(c["source"])
    for key in ["def gradcam", "gradcam_examples.png", "Hardest mistakes", "## Phase 10"]:
        if key in t: src_by_key[key] = c["source"]
patched = 0
for c in ex["cells"]:
    t = "".join(c["source"])
    if c["cell_type"] == "markdown" and "## Phase 10" in t:
        c["source"] = src_by_key["## Phase 10"]; patched += 1
    elif c["cell_type"] == "code" and "def gradcam" in t:
        c["source"] = src_by_key["def gradcam"]; patched += 1
    elif c["cell_type"] == "code" and "gradcam_examples.png" in t:
        c["source"] = src_by_key["gradcam_examples.png"]
        c["outputs"] = [o for o in c["outputs"] if "image/png" not in o.get("data", {})] + [img_output(ART / "gradcam_examples.png")]; patched += 1
    elif c["cell_type"] == "code" and "Hardest mistakes" in t and (ART / "gradcam_errors.png").exists():
        c["source"] = src_by_key["Hardest mistakes"]
        c["outputs"] = [o for o in c["outputs"] if "image/png" not in o.get("data", {})] + [img_output(ART / "gradcam_errors.png")]; patched += 1
json.dump(ex, open(EXEC_NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"Patched {patched} cells. Errors shown: {errs.error.value_counts().to_dict()}")
print("Saved artifacts/gradcam_examples.png and artifacts/gradcam_errors.png")
