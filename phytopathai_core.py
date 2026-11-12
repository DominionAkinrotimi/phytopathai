"""Model, explanations and lesion clustering for PhytopathAI.

Classifier (selected in notebooks/phytopathai_experiments.ipynb): the image is
resized to 1024x768, a 3x4 grid of 224x224 patches is cut from its centre,
each patch goes through ImageNet ResNet50 (frozen), the 12 average-pooled
feature vectors are averaged, standardised, and scored by a logistic
regression. Test accuracy on the 80 held-out BACH images: 85.0%.

Because the classifier is linear in the averaged features, both explanations
are exact:
- class activation map: each patch's 7x7 ResNet50 feature map dotted with the
  class weights shows where inside the patch the evidence for the class sits;
- patch SHAP values (logit scale): with the training-set mean as the baseline,
  the Shapley value of patch p for class c is (1/12) * w_c . (f_p - mean),
  and the values add up exactly to the class logit minus the baseline logit.
"""

from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_prep

CLASS_NAMES = ["Normal", "Benign", "InSitu", "Invasive"]
READABLE = {"Normal": "Normal tissue", "Benign": "Benign lesion",
            "InSitu": "In situ carcinoma", "Invasive": "Invasive carcinoma"}
P = 224
HALF = (1024, 768)          # width, height used in the experiments
ROWS, COLS = 3, 4
HERE = Path(__file__).parent
PATCH_MODEL_FILE = HERE / "app" / "patch_model.npz"
CLUSTER_FILE = HERE / "app" / "cluster_params.npz"


def load_models():
    net = ResNet50(weights="imagenet", include_top=False, pooling=None, input_shape=(P, P, 3))
    patch_model = {k: v for k, v in np.load(PATCH_MODEL_FILE).items()}
    clusters = {k: v for k, v in np.load(CLUSTER_FILE).items()}
    return net, patch_model, clusters


def he_stain_check(rgb):
    """Rough check that an image looks like H&E histology, run before classifying.

    H&E slides are pink/purple tissue on a white background with almost no
    green or blue, and fine texture everywhere. Thresholds were set on 13
    histology images (lowest pink/purple share 0.42, green/blue always 0.00,
    lowest median local texture 0.043) and 36 ordinary images (photos,
    thumbnails, slides, screenshots), all of which fail at least one test.
    It is a heuristic, not a guarantee.
    """
    from matplotlib.colors import rgb_to_hsv

    small = cv2.resize(np.asarray(rgb, dtype=np.uint8), (P, P)) / 255.0
    hsv = rgb_to_hsv(small)
    h, s, v = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    tissue = ~((v > 0.85) & (s < 0.12))
    tissue_share = float(tissue.mean())
    gray = small.mean(axis=-1).astype(np.float32)
    mean = cv2.blur(gray, (5, 5))
    local_sd = np.sqrt(np.clip(cv2.blur(gray * gray, (5, 5)) - mean * mean, 0, None))
    texture = float(np.median(local_sd))
    if tissue_share < 0.05:
        return False, {"tissue": tissue_share, "pink_purple": 0.0, "green_blue": 0.0, "texture": texture}
    n = tissue.sum()
    pink_purple = float((((h >= 260) | (h <= 20)) & (s >= 0.08) & tissue).sum() / n)
    green_blue = float(((h > 70) & (h < 200) & (s > 0.15) & tissue).sum() / n)
    ok = tissue_share >= 0.20 and pink_purple >= 0.35 and green_blue <= 0.10 and texture >= 0.035
    return ok, {"tissue": tissue_share, "pink_purple": pink_purple, "green_blue": green_blue,
                "texture": texture}


def grid_patches(half):
    """Central 3x4 grid of 224x224 patches from a 768x1024 image (same as the experiments)."""
    H, W = half.shape[:2]
    y0, x0 = (H - ROWS * P) // 2, (W - COLS * P) // 2
    crop = half[y0:y0 + ROWS * P, x0:x0 + COLS * P]
    patches = np.stack([crop[P * i:P * (i + 1), P * j:P * (j + 1)]
                        for i in range(ROWS) for j in range(COLS)])
    return patches, crop


def classify(net, pm, rgb):
    """Returns probabilities plus everything the explanations need."""
    half = cv2.resize(np.asarray(rgb, dtype=np.uint8), HALF, interpolation=cv2.INTER_LINEAR)
    patches, crop = grid_patches(half)
    conv = net.predict(resnet_prep(patches.astype("float32")), verbose=0)   # (12, 7, 7, 2048)
    patch_feats = conv.mean(axis=(1, 2))                                     # = ResNet50 avg pooling
    z = (patch_feats.mean(axis=0) - pm["scaler_mean"]) / pm["scaler_scale"]
    logits = z @ pm["coef"].T + pm["intercept"]
    probs = np.exp(logits - logits.max())
    return {"probs": probs / probs.sum(), "logits": logits, "conv": conv,
            "patch_feats": patch_feats, "crop": crop}


def class_weights(pm, c):
    return pm["coef"][c] / pm["scaler_scale"]


def patch_shap(pm, res, c):
    """Exact Shapley value of each patch for class c (logit scale), shape (3, 4)."""
    w = class_weights(pm, c)
    phi = (res["patch_feats"] - pm["scaler_mean"]) @ w / (ROWS * COLS)
    return phi.reshape(ROWS, COLS)


def activation_map(pm, res, c):
    """Class activation map over the 672x896 crop: evidence for class c by location."""
    w = class_weights(pm, c)
    maps = res["conv"] @ w                                                   # (12, 7, 7)
    tiles = [cv2.resize(m.astype(np.float32), (P, P), interpolation=cv2.INTER_CUBIC) for m in maps]
    rows = [np.hstack(tiles[i * COLS:(i + 1) * COLS]) for i in range(ROWS)]
    return np.vstack(rows)


def whole_224(rgb):
    return cv2.resize(np.asarray(rgb, dtype=np.uint8), (P, P), interpolation=cv2.INTER_LINEAR)


def assign_cluster(net, params, img224):
    """Nearest K-Means centre for ResNet50 features of the whole image (as in the pipeline)."""
    feats = net.predict(resnet_prep(img224.astype("float32")[None].copy()), verbose=0)[0].mean(axis=(0, 1))
    scaled = (feats - params["scaler_mean"]) / params["scaler_scale"]
    dists = np.linalg.norm(params["centers"] - scaled, axis=1)
    cid = int(dists.argmin())
    return cid, float(dists[cid]), float(params["ood_threshold_p95"])


def signed_overlay(img, values, alpha=0.55):
    """Red where values are positive, blue where negative, scaled to the strongest value."""
    lim = np.abs(values).max() or 1.0
    v = np.clip(values / lim, -1, 1)
    colour = np.zeros(img.shape, dtype=np.float32)
    colour[..., 0] = np.clip(v, 0, 1) * 255
    colour[..., 2] = np.clip(-v, 0, 1) * 255
    a = (np.abs(v) * alpha)[..., None]
    return (img * (1 - a) + colour * a).astype(np.uint8)


def patch_grid_overlay(crop, phi, alpha=0.5):
    """Colour each of the 12 patches by its SHAP value and draw the grid."""
    big = np.kron(phi, np.ones((P, P)))
    out = signed_overlay(crop, big, alpha)
    for i in range(1, ROWS):
        out[i * P - 1:i * P + 1] = 255
    for j in range(1, COLS):
        out[:, j * P - 1:j * P + 1] = 255
    return out
