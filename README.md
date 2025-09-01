# PhytopathAI: AI-Powered Breast Cancer Histopathology Analysis with Phytotherapeutic Recommendations

![PhytopathAI](https://img.shields.io/badge/BSc-Research-blue)
![License](https://img.shields.io/badge/License-CC--BY--4.0-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)

An end-to-end deep learning system that combines computational pathology with ethnobotanical discovery to provide diagnostic classification, biological subtyping, and evidence-based phytochemical recommendations for breast cancer histopathology images.

## Research Overview

This research project develops a novel framework that extends beyond traditional cancer classification to discover intrinsic biological patterns and connect them to potential plant-derived therapeutics. The system demonstrates how AI can bridge modern computational pathology with traditional medicinal knowledge.

## Key Features

- **Diagnostic Classification**: VGG16-based model achieving 78.75% accuracy in 4-class classification (Normal, Benign, In Situ, Invasive)
- **Biological Subtyping**: Unsupervised discovery of two clinically relevant tumor subtypes using ResNet50 features and K-Means clustering
- **Phytochemical Recommendations**: Evidence-based mapping of tumor biology to plant-derived compounds with known mechanisms
- **Morphological Analysis**: Quantitative assessment of nuclear features and tissue architecture
- **Reproducibility Framework**: Comprehensive validation system for new data assessment

## 📊 Performance Metrics

### Classification Performance
- **Overall Accuracy**: 78.75%
- **Precision (Invasive)**: 85%
- **Recall (Benign)**: 88%
- **F1-Score**: 0.73 (macro average)

### Clustering Results
- **Silhouette Score**: 0.066 (training), 0.045 (validation)
- **Cluster 0**: 94.8% less aggressive tumors (Benign/In Situ)
- **Cluster 1**: 67.4% invasive carcinomas
- **Morphological Significance**: 
  - Glandular organization: p < 0.0001
  - Nuclear pleomorphism: p = 0.037

## Architecture

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6851ff00a4d763150636298d/9eK33GLj7wCpO9zGgZnd9.png)


## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/phytopathai.git
cd phytopathai

# Install dependencies
pip install -r requirements.txt

# Download models (if not included)
# Place models in /models directory
```

### Usage

```python
from app import analyze_breast_cancer

# Analyze a single image
results, explanation = analyze_breast_cancer(your_image_array)
print(explanation)
```

### Web Interface

```bash
# Launch Gradio UI
python app.py

# Access at http://localhost:7860
```

## 📁 Project Structure

```
phytopathai/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── models/               # Trained models
│   ├── classifier_final.h5
│   ├── kmeans_model.pkl
│   ├── feature_scaler.pkl
│   └── validation_assets.pkl
├── results/              # Analysis outputs
│   └── tables/
│       ├── phytochemical_recommendation_db.csv
│       └── classification_report.csv
├── examples/             # Sample images
└── validation_function.py # Reproducibility framework
```

## 🌿 Phytochemical Database

The system includes 20+ evidence-based plant-compound mappings:

- **Cluster 0 (Less Aggressive)**: Genistein (Soy), Resveratrol (Grapes)
- **Cluster 1 (More Aggressive)**: Curcumin (Turmeric), Sulforaphane (Broccoli)
- **Broad Spectrum**: EGCG (Green Tea), Quercetin (Ginkgo)

Each recommendation includes:
- Scientific mechanisms of action
- PubMed references
- Concentration data
- Safety profiles

## 🔬 Methodological Details

### Data
- **Dataset**: BACH ICIAR 2018 (400 H&E-stained images)
- **Classes**: Normal, Benign, In Situ, Invasive
- **Preprocessing**: 224×224 resize, ImageNet normalization, histology-specific augmentation

### Training Strategy
1. **Phase 1**: Feature extraction with frozen VGG16 backbone
2. **Phase 2**: Fine-tuning of final convolutional layers
3. **Validation**: Stratified 80/20 split with early stopping

### Clustering Approach
- **Features**: ResNet50 penultimate layer (2048 dimensions)
- **Algorithm**: K-Means with silhouette analysis
- **Validation**: External validation set and morphological consistency checks

## ⚠️ Important Limitations & Caveats

### Clinical Validation Status
🚨 **This is a research prototype, not a clinical tool.** Important limitations include:

- **No Expert Pathologist Validation**: Cluster interpretations are based on computational morphology without validation by board-certified pathologists
- **Limited Dataset**: Trained on only 400 images from a single dataset (BACH 2018)
- **No IHC Correlation**: Biological subtypes are inferred from H&E morphology alone without immunohistochemical validation
- **Retrospective Analysis**: No prospective clinical validation has been performed
- **Sample Bias**: Potential demographic and preparation biases in training data

### Technical Limitations
- **Moderate Accuracy**: 78.75% accuracy may be insufficient for clinical deployment
- **Feature Stability**: Cluster assignments show some instability (silhouette ~0.05)
- **Generalization Unknown**: Performance on external datasets not extensively validated
- **Computational Requirements**: Requires GPU for optimal performance

### Ethical Considerations
- **Not for Diagnostic Use**: This system should not be used for clinical decision-making
- **Supplementary Tool**: intended as a research aid, not a replacement for pathologist expertise
- **Phytochemical Precautions**: Recommendations are based on computational evidence, not clinical trials

## 🎯 Future Work

- [ ] Multi-institutional validation studies
- [ ] Integration with IHC and genomic data
- [ ] Prospective clinical validation
- [ ] Expansion to other cancer types
- [ ] Mobile deployment for low-resource settings
- [ ] Real-time collaboration with pathologists

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@bscthesis{akinrotimi2024phytopathai,
  title={A Hybrid Deep Learning Framework for Breast Tumor Classification and Phytochemical Recommendation in Histopathology Images},
  author={Akinrotimi, Dominion},
  year={2024},
  school={Olusegun Agagu University of Science and Technology}
}
```

## 📄 License

This project is licensed under the Creative Commons Attribution 4.0 International License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

This is a BSc research project. For research collaborations or questions, please contact the author.

## 📞 Contact

**Dominion Akinrotimi**  
BSc student, Computer Science  
Olusegun Agagu University of Science and Technology  
[Email](akinrotimioyin@gmail.com) | [GitHub](https://github.com/DominionAkinrotimi) | [LinkedIn](https://www.linkedin.com/in/dominion-akinrotimi-7a5961268)

---

**Disclaimer**: This tool is for research purposes only. Not for clinical use. Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.

---
