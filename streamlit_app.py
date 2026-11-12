"""PhytopathAI: breast histology classification with exact explanations, lesion
clustering, and literature-linked phytochemicals. Research prototype."""

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import phytopathai_core as core

st.set_page_config(page_title="PhytopathAI", page_icon="🔬", layout="wide")
R = Path("results")
E = R / "experiments"


@st.cache_resource(show_spinner="Loading the model (first visit only)...")
def models():
    return core.load_models()


@st.cache_data
def tables():
    test = json.loads((E / "test_summary.json").read_text())
    report = pd.read_csv(E / "classification_report_best.csv", index_col=0)
    cv = pd.read_csv(E / "cv_results.csv")
    old = json.loads((R / "training_summary.json").read_text())
    defs = json.loads((R / "cluster_definitions.json").read_text())
    recs = pd.read_csv(R / "phytochemical_recommendations_final.csv")
    return test, report, cv, old, defs, recs


def figure(container, path, caption):
    # Images are embedded as data URIs: Streamlit Cloud's media endpoint returns
    # HTTP 500 for st.image files.
    data = base64.b64encode(Path(path).read_bytes()).decode()
    container.markdown(
        f"<figure style='margin:0'><img src='data:image/png;base64,{data}' style='width:100%'>"
        f"<figcaption style='font-size:0.85rem;color:gray'>{caption}</figcaption></figure>",
        unsafe_allow_html=True)


def show(container, arr, caption, width=560):
    """Show a numpy image inline, keeping its aspect ratio."""
    img = Image.fromarray(arr)
    img = img.resize((width, round(img.height * width / img.width)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode()
    container.markdown(
        f"<figure style='margin:0'><img src='data:image/png;base64,{data}' style='width:100%;"
        f"max-width:{width}px'><figcaption style='font-size:0.85rem;color:gray'>{caption}"
        f"</figcaption></figure>", unsafe_allow_html=True)


test, report, cv, old, defs, recs = tables()
CATEGORY = {"Invasive": "Invasive-enriched", "Benign": "Early-lesion", "InSitu": "Early-lesion"}

st.title("PhytopathAI")
st.caption("Breast histology (H&E) classification with visual explanations, lesion clustering, and "
           "literature-linked plant compounds. **Research prototype from a BSc thesis. Not a medical "
           "device. Nothing here is diagnosis or treatment advice.**")

tab_run, tab_method = st.tabs(["Analyse an image", "Results and method"])

with tab_run:
    st.write("Upload an H&E-stained breast histology microscopy image, like those in the "
             "[BACH 2018 dataset](https://iciar2018-challenge.grand-challenge.org/) the model was "
             "trained on. Images that don't look like H&E histology are refused before analysis.")
    up = st.file_uploader("Image (TIFF, PNG or JPG)", type=["tif", "tiff", "png", "jpg", "jpeg"])
    ok = False

    if up:
        rgb = np.array(Image.open(up).convert("RGB"))
        ok, colour = core.he_stain_check(rgb)
        if not ok:
            show(st, core.whole_224(rgb), "Uploaded image", width=300)
            st.error("This doesn't look like an H&E-stained histology image, so it hasn't been "
                     "analysed. The model only knows four kinds of breast tissue and would give one of "
                     "those answers for any picture, which would be meaningless.")
            st.caption(f"Pink/purple stain: {colour['pink_purple']:.0%} of the image (needs at least "
                       f"35%). Green/blue: {colour['green_blue']:.0%} (needs at most 10%). Non-background: "
                       f"{colour['tissue']:.0%} (needs at least 20%). Fine texture: "
                       f"{colour['texture']:.3f} (needs at least 0.035). This check is a rough filter, "
                       "not a guarantee.")
            ok = st.checkbox("It is an H&E histology image (for example a pale stain). Analyse it anyway.")

    if up and ok:
        net, pm, clusters = models()
        res = core.classify(net, pm, rgb)
        probs = res["probs"]
        k = int(probs.argmax())
        name = core.CLASS_NAMES[k]

        c1, c2 = st.columns([1, 1])
        show(c1, res["crop"], "What the model reads: the image at 1024×768, centre cut into 12 patches")
        with c2:
            st.subheader(f"Prediction: {core.READABLE[name]}")
            st.write(f"Model probability **{probs[k]:.0%}**. On held-out test images this class was "
                     f"correctly recognised {report.loc[name, 'recall']:.0%} of the time.")
            for c, p in zip(core.CLASS_NAMES, probs):
                st.progress(float(p), text=f"{core.READABLE[c]}: {p:.1%}")

        st.subheader("Why the model decided this")
        cam = core.activation_map(pm, res, k)
        phi = core.patch_shap(pm, res, k)
        g1, g2 = st.columns(2)
        show(g1, core.signed_overlay(res["crop"], cam),
             f"Class activation map: red areas are evidence for '{core.READABLE[name]}', "
             "blue areas evidence against.")
        show(g2, core.patch_grid_overlay(res["crop"], phi),
             f"SHAP value of each patch for '{core.READABLE[name]}': red patches pushed the "
             "prediction towards it, blue patches away.")
        top = np.unravel_index(np.argmax(phi), phi.shape)
        st.caption(f"Both are exact for this model, not approximations. The 12 patch SHAP values add "
                   f"up to the class score; the strongest patch is row {top[0] + 1}, column {top[1] + 1}. "
                   "Each patch covers about 448×448 pixels of the original image.")

        if name == "Normal":
            st.info("Predicted normal tissue, so no lesion cluster or compounds are shown.")
        else:
            st.subheader("Lesion cluster")
            cid, dist, limit = core.assign_cluster(net, clusters, core.whole_224(rgb))
            info = defs[str(cid)]
            comp = ", ".join(f"{d} {p:.0f}%" for d, p in info["diagnostic_composition_pct"].items())
            st.write(f"Closest to **cluster {cid} ({CATEGORY[info['dominant_diagnosis']]})**. "
                     f"Training images in this cluster: {comp}.")
            if dist > limit:
                st.warning("This image is further from the cluster centres than 95% of training "
                           "images. Treat the cluster and the compounds below with extra caution.")

            st.subheader("Plant compounds studied against this lesion type")
            st.caption("Laboratory and animal studies only (tier A: cell and animal studies; tier B: "
                       "cell studies). This is a literature map for research, **not a treatment "
                       "recommendation**. Anyone with breast cancer should follow their care team.")
            rows = recs[recs.cluster == cid].sort_values("evidence_tier")
            for _, r in rows.iterrows():
                with st.expander(f"{r.compound} ({r.source_plant}) · tier {r.evidence_tier}"):
                    st.write(f"**Mechanism:** {r.mechanism}")
                    st.write(f"**Why this cluster:** {r.cluster_anchor_rationale}")
                    st.write(f"**Evidence:** {r.study_design}")
                    st.write(f"**Source:** [{r.citation_title}]({r.pubmed_url})")

with tab_method:
    lo, hi = test["test_accuracy_95ci"]
    sel = test["selected"]
    st.subheader("Classifier")
    st.write("Each image is resized to 1024×768 and its centre cut into 12 patches of 224×224. "
             "ImageNet ResNet50 (frozen) turns each patch into 2,048 features; the 12 are averaged and "
             "a logistic regression assigns the class. This design was chosen from 20 candidates "
             "(5 networks × whole image or patches × with or without stain normalisation) by 5-fold "
             "cross-validation on 320 images, then tested once on 80 held-out images.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test accuracy", f"{test['test_accuracy']:.1%}", help=f"95% CI {lo:.0%} to {hi:.0%}")
    m2.metric("Macro F1", f"{test['test_macro_f1']:.2f}")
    m3.metric("Matthews correlation", f"{test['test_mcc']:.2f}")
    m4.metric("Cross-validation", f"{sel['cv_accuracy']:.1%}", help=f"± {sel['cv_accuracy_sd']:.1%} across 5 folds")
    st.caption(f"95% bootstrap confidence interval for test accuracy: {lo:.0%} to {hi:.0%}. The earlier "
               f"fine-tuned VGG16 on whole images scored {old['test_accuracy']:.1%} on the same 80 images "
               f"(McNemar exact p = {test['mcnemar']['exact_p']:.3f}).")
    c1, c2 = st.columns(2)
    figure(c1, E / "cv_comparison.png", "All 20 candidates, 5-fold cross-validation")
    figure(c2, E / "confusion_matrix_best.png", "Selected model, 80 test images")
    st.write("Cutting the image into higher-resolution patches helped every network. Network size did "
             "not: a 7 M-parameter DenseNet121 came within 1.3 points of the best, and VGG16 (14.7 M) "
             "was the weakest.")
    st.subheader("Clusters")
    v = json.loads((R / "validation_summary.json").read_text())
    st.write(f"ResNet50 features of benign, in situ and invasive images, grouped with K-Means (k=2). "
             f"Silhouette {v['train_silhouette']:.2f} (training) and {v['heldout_silhouette']:.2f} "
             f"(held-out): the groups overlap a lot. They still line up with diagnosis: one holds most "
             f"invasive images (held-out chi-square p = {v['heldout_chi2_p']:.0e}).")
    figure(st, R / "tsne_clusters_vs_diagnosis.png", "Lesion images: K-Means clusters (left) vs true diagnosis (right)")
    st.subheader("Limitations")
    st.write("- 400 images from one dataset; no external validation.\n"
             "- No pathologist reviewed the clusters or the explanation maps.\n"
             "- Compound links come from published lab studies, not from this model's data, and are "
             "not evidence that any compound helps patients.")
