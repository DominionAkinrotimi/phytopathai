"""Writes notebooks/phytopathai_experiments.ipynb (run: python notebooks/make_experiments_nb.py)."""
import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip()})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                               "outputs": [], "source": s.strip()})

md("""
# PhytopathAI experiments: does a bigger network, patches or stain normalisation help?

Runtime → T4 GPU, then Run all (about 20–30 minutes). Results go to `MyDrive/BreastCancerProject/rebuild_v3/experiments/` and download as `experiment_results.zip`.

## Protocol (written before any result was seen)
1. **Test set:** the same 80 images as the rebuilt classifier (same seed and stratified split). Locked until step 5 and used once.
2. **Development set:** the other 320 images. All comparisons use 5-fold stratified cross-validation on these only.
3. **Variants (20):** 5 ImageNet backbones of different size (EfficientNetB0, DenseNet121, VGG16, ResNet50, ConvNeXt-Tiny) × input (whole image at 224×224, or 12 patches of 224×224 from the image at half resolution, features averaged) × stain (none, or Reinhard normalisation to the development-set average).
4. **Model per variant:** frozen backbone features → standardise → logistic regression, with its penalty chosen by an inner 3-fold CV.
5. **Selection:** the variant with the best mean CV accuracy is refitted on all 320 images and scored once on the 80 test images (accuracy with bootstrap 95% CI, macro F1, exact McNemar test against the rebuilt fine-tuned VGG16, which scored 68.75%).

Stain targets are computed from development images only, so nothing about the test images enters any choice.
""")

code("""
import os, json, time, shutil, gc
import numpy as np, pandas as pd, cv2, matplotlib.pyplot as plt
from google.colab import drive, files
drive.mount("/content/drive")
import tensorflow as tf
from tensorflow.keras import applications as A
print("TensorFlow", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU") or "NONE - switch to T4")

PROJECT = "/content/drive/MyDrive/BreastCancerProject"
RAW = f"{PROJECT}/Train"
BASE_RESULTS = f"{PROJECT}/rebuild_v3/results"
OUT = f"{PROJECT}/rebuild_v3/experiments"; os.makedirs(OUT, exist_ok=True)
CLASSES = ["Normal", "Benign", "InSitu", "Invasive"]
SEED = 42; P = 224
""")

md("## 1. Load images: whole (224×224) and 12 patches from the half-resolution image")
code("""
from sklearn.model_selection import train_test_split

def grid_patches(img):  # img is 768 x 1024; central 3 x 4 grid of 224 x 224 tiles
    H, W = img.shape[:2]
    y0, x0 = (H - 3 * P) // 2, (W - 4 * P) // 2
    return np.stack([img[y0 + P * i:y0 + P * (i + 1), x0 + P * j:x0 + P * (j + 1)]
                     for i in range(3) for j in range(4)])

whole, half, labels, names = [], [], [], []
for lab, c in enumerate(CLASSES):
    for fn in sorted(os.listdir(f"{RAW}/{c}")):
        if fn.lower().endswith((".tif", ".tiff")):
            rgb = cv2.cvtColor(cv2.imread(f"{RAW}/{c}/{fn}"), cv2.COLOR_BGR2RGB)
            whole.append(cv2.resize(rgb, (P, P)))          # same as the rebuilt pipeline
            half.append(cv2.resize(rgb, (1024, 768)))
            labels.append(lab); names.append(f"{c}/{fn}")
whole, half, y, names = np.array(whole), np.array(half), np.array(labels), np.array(names)

idx = np.arange(len(y))
dev, test = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
base = pd.read_csv(f"{BASE_RESULTS}/test_predictions.csv")
assert list(base.file) == list(names[test]), "test split differs from the rebuilt pipeline"
print(f"development {len(dev)} | test {len(test)} (same 80 as before)")
""")

md("## 2. Reinhard stain normalisation (targets from development images only)")
code("""
def lab_stats(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return lab.mean(0), lab.std(0) + 1e-6

stats = np.array([np.concatenate(lab_stats(half[i])) for i in dev])
T_MEAN, T_STD = stats[:, :3].mean(0), stats[:, 3:].mean(0)

def reinhard(img):
    m, s = lab_stats(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = (lab - m) / s * T_STD + T_MEAN
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

half_norm = np.array([reinhard(im) for im in half])
whole_norm = np.array([cv2.resize(im, (P, P)) for im in half_norm])
INPUTS = {
    ("whole", "none"): whole[:, None], ("whole", "reinhard"): whole_norm[:, None],
    ("patches", "none"): np.array([grid_patches(im) for im in half]),
    ("patches", "reinhard"): np.array([grid_patches(im) for im in half_norm]),
}
del half_norm; gc.collect()
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].imshow(half[test[0]]); ax[0].set_title("original"); ax[1].imshow(reinhard(half[test[0]])); ax[1].set_title("Reinhard")
[a.axis("off") for a in ax]; plt.savefig(f"{OUT}/stain_example.png", dpi=120, bbox_inches="tight"); plt.show()
""")

md("## 3. Frozen features from 5 backbones")
code("""
BACKBONES = {
    "EfficientNetB0": (A.EfficientNetB0, A.efficientnet.preprocess_input),
    "DenseNet121": (A.DenseNet121, A.densenet.preprocess_input),
    "VGG16": (A.VGG16, A.vgg16.preprocess_input),
    "ResNet50": (A.ResNet50, A.resnet50.preprocess_input),
    "ConvNeXtTiny": (A.ConvNeXtTiny, A.convnext.preprocess_input),
}
FEATS, PARAMS = {}, {}
for bname, (ctor, prep) in BACKBONES.items():
    t0 = time.time()
    net = ctor(weights="imagenet", include_top=False, pooling="avg", input_shape=(P, P, 3))
    PARAMS[bname] = int(net.count_params())
    for (inp, stain), X in INPUTS.items():
        n, k = X.shape[:2]
        flat = X.reshape(n * k, P, P, 3)
        out = [net.predict(prep(flat[i:i + 512].astype("float32")), batch_size=64, verbose=0)
               for i in range(0, len(flat), 512)]
        FEATS[(bname, inp, stain)] = np.concatenate(out).reshape(n, k, -1).mean(1)
    print(f"{bname}: {PARAMS[bname] / 1e6:.1f} M parameters, {time.time() - t0:.0f}s")
    del net; tf.keras.backend.clear_session(); gc.collect()
np.savez_compressed(f"{OUT}/features.npz", **{"|".join(k): v for k, v in FEATS.items()})
""")

md("## 4. 5-fold cross-validation on the 320 development images (test images not used)")
code("""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate

def make_model():
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    return GridSearchCV(pipe, {"logisticregression__C": [0.001, 0.01, 0.1, 1.0]},
                        cv=StratifiedKFold(3, shuffle=True, random_state=SEED), n_jobs=-1)

outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows = []
for (bname, inp, stain), F in FEATS.items():
    r = cross_validate(make_model(), F[dev], y[dev], cv=outer, scoring=["accuracy", "f1_macro"])
    rows.append({"backbone": bname, "params_M": round(PARAMS[bname] / 1e6, 1), "input": inp, "stain": stain,
                 "cv_accuracy": r["test_accuracy"].mean(), "cv_accuracy_sd": r["test_accuracy"].std(),
                 "cv_macro_f1": r["test_f1_macro"].mean()})
cv = pd.DataFrame(rows).sort_values(["cv_accuracy", "params_M"], ascending=[False, True]).reset_index(drop=True)
cv.round(4).to_csv(f"{OUT}/cv_results.csv", index=False)
print(cv.round(3).to_string())

fig, ax = plt.subplots(figsize=(9, 6))
lbl = cv.backbone + " · " + cv.input + " · " + cv.stain
ax.barh(lbl[::-1], cv.cv_accuracy[::-1], xerr=cv.cv_accuracy_sd[::-1], color="#2a7f86")
ax.axvline(0.6875, color="black", ls="--", label="rebuilt fine-tuned VGG16 (test 68.75%)")
ax.set_xlabel("5-fold CV accuracy (development set, mean ± SD)"); ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(f"{OUT}/cv_comparison.png", dpi=200); plt.show()
""")

md("## 5. The selected variant, scored once on the 80 test images")
code("""
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef
from scipy.stats import binomtest

best = cv.iloc[0]
key = (best.backbone, best.input, best.stain)
final = make_model().fit(FEATS[key][dev], y[dev])
pred = final.predict(FEATS[key][test]); yt = y[test]
acc = float((pred == yt).mean())
rng = np.random.default_rng(SEED)
boot = [(pred[s] == yt[s]).mean() for s in (rng.integers(0, len(yt), len(yt)) for _ in range(2000))]
lo, hi = np.percentile(boot, [2.5, 97.5])

old_ok = (base.pred.values == base.true.values); new_ok = (pred == yt)
b, c = int((old_ok & ~new_ok).sum()), int((~old_ok & new_ok).sum())
p_mcnemar = binomtest(c, b + c, 0.5).pvalue if b + c else 1.0

ConfusionMatrixDisplay(confusion_matrix(yt, pred), display_labels=CLASSES).plot(cmap="Blues", values_format="d")
plt.title(f"Test set: {best.backbone}, {best.input}, {best.stain}")
plt.savefig(f"{OUT}/confusion_matrix_best.png", dpi=200, bbox_inches="tight"); plt.show()
pd.DataFrame(classification_report(yt, pred, target_names=CLASSES, output_dict=True)).T.round(4) \\
  .to_csv(f"{OUT}/classification_report_best.csv")
pd.DataFrame({"file": names[test], "true": yt, "pred": pred}).to_csv(f"{OUT}/test_predictions_best.csv", index=False)

summary = {
    "selected": {"backbone": best.backbone, "input": best.input, "stain": best.stain,
                 "params_M": float(best.params_M), "chosen_C": final.best_params_["logisticregression__C"],
                 "cv_accuracy": float(best.cv_accuracy), "cv_accuracy_sd": float(best.cv_accuracy_sd)},
    "test_accuracy": acc, "test_accuracy_95ci": [float(lo), float(hi)],
    "test_macro_f1": float(classification_report(yt, pred, output_dict=True)["macro avg"]["f1-score"]),
    "test_mcc": float(matthews_corrcoef(yt, pred)),
    "baseline_test_accuracy": float(old_ok.mean()),
    "mcnemar": {"baseline_right_new_wrong": b, "baseline_wrong_new_right": c, "exact_p": float(p_mcnemar)},
}
json.dump(summary, open(f"{OUT}/test_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
print(classification_report(yt, pred, target_names=CLASSES, digits=4))
""")

md("## 6. Download")
code("""
# features.npz stays in Drive (large); everything else goes in the zip
tmp ="/content/experiment_results"; shutil.rmtree(tmp, ignore_errors=True)
shutil.copytree(OUT, tmp, ignore=shutil.ignore_patterns("features.npz"))
files.download(shutil.make_archive("/content/experiment_results", "zip", tmp))
""")

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 0,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
out = Path(__file__).with_name("phytopathai_experiments.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
