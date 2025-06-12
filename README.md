# XPLIA: Librairie d'Explicabilité d'IA Complète et Avancée

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://xplia.readthedocs.io)

## Vision

XPLIA est une librairie Python open-source révolutionnaire conçue pour rendre les modèles d'intelligence artificielle plus transparents, interprétables et conformes aux réglementations. Notre mission est de combler le fossé entre la complexité des modèles d'IA modernes et la nécessité de comprendre et d'expliquer leurs décisions.

## Caractéristiques principales

### 🔍 Explicabilité multiméthode
- Intégration harmonieuse des méthodes SHAP, LIME, Anchors, InterpretML et plus
- Support pour tout type de modèle ML/DL (scikit-learn, TensorFlow, PyTorch, etc.)
- Explicabilité locale et globale dans un framework unifié

### 📊 Visualisations interactives de pointe
- Tableaux de bord interactifs avec Dash et Plotly
- Graphiques dynamiques d'influence des caractéristiques
- Visualisation des processus de décision et des chemins d'activation

### 🎯 Adaptabilité multi-audience
- Interface technique pour les data scientists
- Mode "vulgarisé" pour les décideurs et le grand public
- Personnalisation des niveaux de détail et de complexité

### 🔒 Conformité réglementaire intégrée
- Support pour les exigences de l'AI Act européen
- Documentation automatisée pour la conformité RGPD
- Traçabilité et auditabilité des modèles

### 📝 Documentation approfondie
- Documentation des sources de données
- Méthodologies d'entraînement et de validation
- Rapports de conformité automatisés

### 🔄 Architecture modulaire et extensible
- API unifiée et cohérente
- Extensibilité par plugins
- Support continu pour les nouvelles méthodes d'explicabilité

## Installation

```bash
pip install xplia
```

Pour les fonctionnalités avancées:
```bash
pip install xplia[full]
```

## Utilisation rapide

```python
import xplia as xplia

# Charger un modèle pré-entraîné
model = iai.load_model("my_model.pkl")

# Créer un explainer adapté au modèle
explainer = iai.create_explainer(model, method="unified")

# Générer des explications
explanations = explainer.explain(X_test, y_test)

# Visualiser les résultats avec un tableau de bord interactif
iai.visualize.dashboard(explanations, level="technical")
```

## Documentation

Pour une documentation complète, des tutoriels et des exemples, consultez [notre site de documentation](https://xplia.readthedocs.io).

## Contribution

Nous accueillons avec enthousiasme les contributions de la communauté ! Consultez notre [guide de contribution](CONTRIBUTING.md) pour commencer.

## License

Ce projet est sous licence [MIT](LICENSE).

## Citation

Si vous utilisez XPLIA dans votre recherche, veuillez nous citer :
```
@software{xplia2025,
  author = {Severino, Nicolas et al.},
  title = {XPLIA: A Comprehensive AI Model Explainability Framework},
  url = {https://github.com/nseverino/xplia},
  version = {0.1.0},
  year = {2025},
}
```
