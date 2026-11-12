"""Model loading, prediction, explanations and lesion clustering for PhytopathAI.

The classifier is the VGG16 model trained by notebooks/phytopathai_pipeline.ipynb.
Its weights are stored in half precision (39 MB) on Google Drive and downloaded
on first use; on the 80 held-out test images they give the same predictions as
the full-precision model.
"""

from pathlib import Path
import urllib.request

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_prep
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_prep
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model

CLASS_NAMES = ["Normal", "Benign", "InSitu", "Invasive"]
READABLE = {"Normal": "Normal tissue", "Benign": "Benign lesion",
            "InSitu": "In situ carcinoma", "Invasive": "Invasive carcinoma"}
IMG = 224
WEIGHTS_ID = "1dXFXdzlZ-aJi_2l64Q98IFSB4LKsHLzT"
WEIGHTS_URL = f"https://drive.usercontent.google.com/download?id={WEIGHTS_ID}&export=download&confirm=t"
HERE = Path(__file__).parent
WEIGHTS_FILE = HERE / "app" / "phytopathai_vgg16_fp16.npz"
CLUSTER_FILE = HERE / "app" / "cluster_params.npz"
LAST_CONV = "block5_conv3"


def build_classifier():
    """Same architecture as training: VGG16 base + Flatten + Dense(256) + Dropout + Dense(4)."""
    base = VGG16(weights=None, include_top=False, input_tensor=Input(shape=(IMG, IMG, 3)))
    x = Dropout(0.5)(Dense(256, activation="relu")(Flatten()(base.output)))
    return Model(base.input, Dense(4, activation="softmax")(x))


def load_classifier():
    if not WEIGHTS_FILE.exists():
        WEIGHTS_FILE.parent.mkdir(exist_ok=True)
        tmp = WEIGHTS_FILE.with_suffix(".part")
        urllib.request.urlretrieve(WEIGHTS_URL, tmp)
        tmp.rename(WEIGHTS_FILE)
    arrays = np.load(WEIGHTS_FILE)
    weights = [arrays[f"arr_{i}"].astype(np.float32) for i in range(len(arrays.files))]
    model = build_classifier()
    model.set_weights(weights)  # raises if any shape differs from training
    return model


def load_cluster_model():
    p = np.load(CLUSTER_FILE)
    extractor = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    return extractor, {k: p[k] for k in p.files}


def to_model_input(rgb):
    """Whole image resized to 224x224, as in training (no cropping or patches)."""
    return cv2.resize(np.asarray(rgb, dtype=np.uint8), (IMG, IMG), interpolation=cv2.INTER_LINEAR)


def predict(model, img224):
    x = vgg_prep(img224.astype("float32")[None].copy())
    return model.predict(x, verbose=0)[0]


def grad_cam(model, img224, class_idx):
    """Grad-CAM on the last VGG16 convolution layer, scaled to 0-1 at 224x224."""
    grad_model = Model(model.inputs, [model.get_layer(LAST_CONV).output, model.output])
    x = tf.convert_to_tensor(vgg_prep(img224.astype("float32")[None].copy()))
    with tf.GradientTape() as tape:
        conv, preds = grad_model(x, training=False)
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * weights[:, None, None, :], axis=-1))[0].numpy()
    cam = cv2.resize(cam, (IMG, IMG))
    return cam / cam.max() if cam.max() > 0 else cam


def shap_map(model, img224, class_idx, max_evals=400):
    """SHAP values for one class with a blurring image masker (model-agnostic).

    Returns a 224x224 map: positive = region pushed the prediction towards
    class_idx, negative = pushed it away.
    """
    import shap

    def f(batch):
        return model.predict(vgg_prep(np.asarray(batch, dtype="float32").copy()), verbose=0)

    masker = shap.maskers.Image("blur(32,32)", (IMG, IMG, 3))
    explainer = shap.Explainer(f, masker, output_names=CLASS_NAMES)
    sv = explainer(img224[None].astype("float32"), max_evals=max_evals, batch_size=50,
                   outputs=[class_idx], silent=True)
    return sv.values[0, ..., 0].sum(axis=-1)


def assign_cluster(extractor, params, img224):
    feats = extractor.predict(resnet_prep(img224.astype("float32")[None].copy()), verbose=0)[0]
    scaled = (feats - params["scaler_mean"]) / params["scaler_scale"]
    dists = np.linalg.norm(params["centers"] - scaled, axis=1)
    cid = int(dists.argmin())
    return cid, float(dists[cid]), float(params["ood_threshold_p95"])


def overlay(img224, heat, cmap="jet", alpha=0.45):
    """Blend a 0-1 heat map over the image."""
    import matplotlib
    colour = (matplotlib.colormaps[cmap](heat)[..., :3] * 255).astype(np.uint8)
    return (img224 * (1 - alpha) + colour * alpha).astype(np.uint8)


def signed_overlay(img224, values, alpha=0.55):
    """Red where SHAP is positive, blue where negative, scaled to the strongest value."""
    lim = np.abs(values).max() or 1.0
    v = np.clip(values / lim, -1, 1)
    colour = np.zeros_like(img224, dtype=np.float32)
    colour[..., 0] = np.clip(v, 0, 1) * 255
    colour[..., 2] = np.clip(-v, 0, 1) * 255
    a = (np.abs(v) * alpha)[..., None]
    return (img224 * (1 - a) + colour * a).astype(np.uint8)
