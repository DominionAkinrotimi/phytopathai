"""PhytopathAI: breast histology classification with Grad-CAM/SHAP explanations,
lesion clustering, and literature-linked phytochemicals. Research prototype."""

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import phytopathai_core as core

st.set_page_config(page_title="PhytopathAI", page_icon="🔬", layout="wide")
R = Path("results")


@st.cache_resource(show_spinner="Loading the classifier (first visit only)...")
def classifier():
    return core.load_classifier()


@st.cache_resource(show_spinner="Loading the clustering model (first lesion only)...")
def cluster_model():
    return core.load_cluster_model()


@st.cache_data
def tables():
    summary = json.loads((R / "training_summary.json").read_text())
    defs = json.loads((R / "cluster_definitions.json").read_text())
    recs = pd.read_csv(R / "phytochemical_recommendations_final.csv")
    report = pd.read_csv(R / "classification_report.csv", index_col=0)
    return summary, defs, recs, report


def figure(container, name, caption):
    data = base64.b64encode((R / name).read_bytes()).decode()
    container.markdown(
        f"<figure style='margin:0'><img src='data:image/png;base64,{data}' style='width:100%'>"
        f"<figcaption style='font-size:0.85rem;color:gray'>{caption}</figcaption></figure>",
        unsafe_allow_html=True)


summary, defs, recs, report = tables()
CATEGORY = {"Invasive": "Invasive-enriched", "Benign": "Early-lesion", "InSitu": "Early-lesion"}

st.title("PhytopathAI")
st.caption("Breast histology (H&E) classification with visual explanations, lesion clustering, and "
           "literature-linked plant compounds. **Research prototype from a BSc thesis. Not a medical "
           "device. Nothing here is diagnosis or treatment advice.**")

tab_run, tab_method = st.tabs(["Analyse an image", "Results and method"])

with tab_run:
    st.write("Upload an H&E-stained breast histology microscopy image, like those in the "
             "[BACH 2018 dataset](https://iciar2018-challenge.grand-challenge.org/) the model was "
             "trained on. Other kinds of images will still get an answer, but it will mean nothing.")
    up = st.file_uploader("Image (TIFF, PNG or JPG)", type=["tif", "tiff", "png", "jpg", "jpeg"])

    if up:
        rgb = np.array(Image.open(up).convert("RGB"))
        img = core.to_model_input(rgb)
        model = classifier()
        probs = core.predict(model, img)
        k = int(probs.argmax())
        name = core.CLASS_NAMES[k]

        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(img, caption="What the model sees (whole image resized to 224×224)",
                     width="stretch")
        with c2:
            st.subheader(f"Prediction: {core.READABLE[name]}")
            st.write(f"Model probability **{probs[k]:.0%}**. On held-out test images this class was "
                     f"correctly recognised {report.loc[name, 'recall']:.0%} of the time.")
            st.bar_chart(pd.DataFrame({"probability": probs},
                                      index=[core.READABLE[c] for c in core.CLASS_NAMES]))

        st.subheader("Why the model decided this")
        g1, g2 = st.columns(2)
        with g1:
            cam = core.grad_cam(model, img, k)
            st.image(core.overlay(img, cam), width="stretch",
                     caption=f"Grad-CAM: warm colours are the regions that most raised the "
                             f"'{core.READABLE[name]}' score.")
        with g2:
            if st.button("Run SHAP (slower, about a minute)"):
                with st.spinner("Computing SHAP values..."):
                    sv = core.shap_map(model, img, k)
                st.image(core.signed_overlay(img, sv), width="stretch",
                         caption=f"SHAP: red regions pushed towards '{core.READABLE[name]}', "
                                 "blue regions pushed away.")
            else:
                st.info("SHAP gives a second, independent explanation. Where it agrees with "
                        "Grad-CAM, the evidence that the model used that region is stronger.")
        st.caption("Both maps work at 224×224, so they point to areas of tissue, not individual cells.")

        if name == "Normal":
            st.info("Predicted normal tissue, so no lesion cluster or compounds are shown.")
        else:
            st.subheader("Lesion cluster")
            extractor, params = cluster_model()
            cid, dist, limit = core.assign_cluster(extractor, params, img)
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
    lo, hi = summary["accuracy_95ci"]
    st.subheader("Classifier")
    st.write(f"VGG16 pretrained on ImageNet, fine-tuned on the 400-image BACH dataset (4 classes, 100 "
             f"each). Split: {summary['train_images']} training, {summary['validation_images']} "
             f"validation, {summary['test_images']} test images. The test images were used once, after "
             f"all choices were made.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test accuracy", f"{summary['test_accuracy']:.1%}", help=f"95% CI {lo:.0%} to {hi:.0%}")
    m2.metric("Macro F1", f"{summary['macro_f1']:.2f}")
    m3.metric("Matthews correlation", f"{summary['matthews_corrcoef']:.2f}")
    st.caption(f"95% bootstrap confidence interval for accuracy: {lo:.0%} to {hi:.0%}. The original "
               "thesis reported 78.75%, but that run picked its best epoch using the test images and "
               "skipped VGG16's input preprocessing. Both are fixed here.")
    c1, c2 = st.columns(2)
    figure(c1, "confusion_matrix.png", "Test set (80 images)")
    figure(c2, "tsne_clusters_vs_diagnosis.png",
           "Lesion images: K-Means clusters (left) vs true diagnosis (right)")
    st.subheader("Clusters")
    v = json.loads((R / "validation_summary.json").read_text())
    st.write(f"ResNet50 features of benign, in situ and invasive images, grouped with K-Means (k=2). "
             f"Silhouette {v['train_silhouette']:.2f} (training) and {v['heldout_silhouette']:.2f} "
             f"(held-out): the groups overlap a lot. They still line up with diagnosis: one holds most "
             f"invasive images (held-out chi-square p = {v['heldout_chi2_p']:.0e}).")
    st.subheader("Limitations")
    st.write("- 400 images from one dataset; no external validation.\n"
             "- Whole images shrunk to 224×224, which loses cell-level detail.\n"
             "- No pathologist reviewed the clusters or the explanation maps.\n"
             "- Compound links come from published lab studies, not from this model's data, and are "
             "not evidence that any compound helps patients.")
