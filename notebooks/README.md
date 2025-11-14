# 📚 XPLIA Interactive Notebooks

Welcome to XPLIA's collection of interactive Jupyter notebooks! These notebooks provide hands-on tutorials and examples for all features of XPLIA.

## 🚀 Getting Started

### Prerequisites

```bash
# Install XPLIA with full dependencies
pip install xplia[full]

# Install Jupyter
pip install jupyter
```

### Launch Jupyter

```bash
cd notebooks
jupyter notebook
```

## 📓 Available Notebooks

### 1. **01_getting_started.ipynb** ⭐ START HERE!

**Perfect for: First-time users**

Learn the basics of XPLIA in 30 minutes:
- Installation and setup
- Basic explanation generation
- Visualization of explanations
- Understanding quality metrics
- Comparing multiple explanation methods

**Key Concepts:**
- SHAP, LIME, Unified explainers
- Feature importance
- Quality metrics
- Interactive visualizations

---

### 2. **02_tier1_advanced_features.ipynb** 🔥 ADVANCED

**Perfect for: Users wanting cutting-edge features**

Explore TIER 1 advanced features that NO other library has:

**Covered Topics:**
1. 🎨 **Multimodal AI**
   - CLIP (Vision-Language models)
   - Stable Diffusion explanations
   - Cross-modal attention analysis

2. 🕸️ **Graph Neural Networks**
   - GNN explainability
   - Molecular explainability for drug discovery
   - Toxicity prediction

3. 🎮 **Reinforcement Learning**
   - Policy explanations
   - Q-value decomposition
   - Trajectory analysis

4. 📈 **Time Series**
   - Temporal importance
   - Forecast explanations
   - Anomaly detection with reasons

5. 🎭 **Generative Models**
   - VAE latent space analysis
   - GAN generator explanations
   - StyleGAN W-space disentanglement

**Prerequisites:** Basic understanding of XPLIA (complete notebook 01 first)

---

### 3. **03_compliance_gdpr_ai_act.ipynb** 🏛️ COMPLIANCE

**Perfect for: Enterprise users, regulated industries**

**THE ONLY XAI library with built-in regulatory compliance!**

**Covered Topics:**
1. **GDPR Compliance**
   - Right to Explanation (Articles 13-15)
   - DPIA Generation (Article 35)
   - Automated PDF reports for auditors

2. **EU AI Act Compliance**
   - Automatic risk category assessment
   - Documentation generation
   - HIGH RISK use case handling (credit, healthcare, etc.)

3. **HIPAA Compliance** (Healthcare)
   - Audit trail logging
   - Patient data access tracking

4. **Fairwashing Detection** ⚠️ UNIQUE!
   - Detect deceptive explanations
   - Identify bias hiding
   - Trust evaluation

5. **Uncertainty Quantification**
   - Epistemic vs Aleatoric uncertainty
   - Confidence intervals
   - High-uncertainty case flagging

**Use Cases:**
- 🏦 Finance (Credit scoring, Insurance, Trading)
- 🏥 Healthcare (Diagnosis, Treatment planning)
- ⚖️ Legal Tech
- 🏛️ Government services

**Prerequisites:** Basic understanding of XPLIA

---

## 🎯 Recommended Learning Path

### For Beginners:
1. Start with **01_getting_started.ipynb**
2. Then explore **03_compliance_gdpr_ai_act.ipynb** if in regulated industry
3. Finally, dive into **02_tier1_advanced_features.ipynb** for cutting-edge features

### For Advanced Users:
1. Skim **01_getting_started.ipynb** for syntax reference
2. Jump to **02_tier1_advanced_features.ipynb** for advanced features
3. Review **03_compliance_gdpr_ai_act.ipynb** for production deployment

### For Compliance Officers:
1. Go directly to **03_compliance_gdpr_ai_act.ipynb**
2. Review **01_getting_started.ipynb** for basic concepts

---

## 📊 Datasets Used

All notebooks use either:
- **Built-in datasets** (Iris, make_classification) - no download needed
- **Synthetic data** - generated on-the-fly
- **Example data** - small CSV files included

**No external data download required!** All notebooks run out-of-the-box.

---

## 🔧 Troubleshooting

### Notebook kernel crashes

```bash
# Restart kernel: Kernel > Restart
# Or restart Jupyter entirely
```

### Missing dependencies

```bash
# Install full XPLIA
pip install xplia[full]

# Or specific dependencies
pip install numpy pandas scikit-learn matplotlib plotly
```

### ImportError

Make sure XPLIA is installed:
```bash
pip install -e ".[full]"  # If running from source
```

### SHAP/LIME not working

```bash
pip install shap lime
```

---

## 💡 Tips for Best Experience

1. **Run cells sequentially** - Don't skip cells!
2. **Restart kernel if errors** - Kernel > Restart & Clear Output
3. **Modify and experiment** - Change parameters and see what happens
4. **Save your work** - File > Save and Checkpoint
5. **Export results** - All notebooks generate HTML/PDF reports you can share

---

## 🚀 Next Steps

After completing the notebooks:

1. **Explore Examples** - Check out `examples/` directory for production-ready code
2. **Read Documentation** - `USAGE_GUIDE_FRANCAIS.md` for comprehensive guide
3. **Try CLI** - `python quickstart.py` for command-line usage
4. **Build Your Own** - Use XPLIA in your projects!

---

## 📞 Support

- **Documentation**: https://xplia.readthedocs.io
- **GitHub Issues**: https://github.com/nicolasseverino/xplia/issues
- **Email**: contact@xplia.com

---

## 🎓 Additional Resources

### Official Guides
- `README.md` - Overview and installation
- `USAGE_GUIDE_FRANCAIS.md` - Complete usage guide (French)
- `COMPLETENESS_ANALYSIS.md` - Feature completeness analysis
- `ARCHITECTURE.md` - Technical architecture

### Examples
- `examples/loan_approval_system.py` - Complete credit scoring system
- `examples/tier1_advanced_features_demo.py` - TIER 1 features demo
- `examples/tier2_tier3_advanced_demo.py` - TIER 2+3 features demo

### Interactive
- `quickstart.py` - Interactive menu for getting started

---

**Happy Learning with XPLIA! 🎉**

*The most comprehensive, production-ready XAI library in the world.*
