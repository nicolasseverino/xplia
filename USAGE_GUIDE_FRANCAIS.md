# 🚀 XPLIA - Guide d'Utilisation Pratique

## Table des Matières
1. [Installation](#installation)
2. [Démarrage Rapide](#démarrage-rapide)
3. [Cas d'Usage par Domaine](#cas-dusage-par-domaine)
4. [Guide par Type de Modèle](#guide-par-type-de-modèle)
5. [Fonctionnalités Avancées](#fonctionnalités-avancées)
6. [Résolution de Problèmes](#résolution-de-problèmes)

---

## 🔧 Installation

### Installation Minimale (Rapide - ~200MB)
```bash
pip install xplia
```

### Installation Complète (Recommandée - ~2GB)
```bash
pip install xplia[full]
```

### Installation depuis le code source (Développement)
```bash
git clone https://github.com/nicolasseverino/xplia.git
cd xplia
pip install -e ".[full]"
```

### Vérification de l'installation
```bash
python -c "import xplia; print(xplia.__version__)"
```

---

## ⚡ Démarrage Rapide

### Exemple Minimal (30 secondes)

```python
from xplia import create_explainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd

# 1. Charger les données
X, y = load_iris(return_X_y=True)
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X_df = pd.DataFrame(X, columns=feature_names)

# 2. Entraîner un modèle
model = RandomForestClassifier(random_state=42)
model.fit(X_df, y)

# 3. Créer l'explainer (XPLIA détecte automatiquement la meilleure méthode)
explainer = create_explainer(model, method='shap', background_data=X_df[:100])

# 4. Expliquer une prédiction
explanation = explainer.explain(X_df[:5])

# 5. Voir les résultats
print("Importance des features:")
print(explanation.feature_importance)

print("\nQualité de l'explication:")
print(explanation.quality_metrics)
```

---

## 🎯 Cas d'Usage par Domaine

### 1. Finance / Banque - Approbation de Crédit

```python
from xplia import create_explainer
from xplia.compliance import GDPRCompliance, AIActCompliance
from xplia.explainers.trust import UncertaintyQuantifier, FairwashingDetector
import pandas as pd

# Vos données
X_train = pd.read_csv('credit_data_train.csv')
y_train = pd.read_csv('credit_labels_train.csv')
X_test = pd.read_csv('credit_data_test.csv')

# Votre modèle (n'importe quel modèle sklearn, XGBoost, etc.)
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)

# 1. Créer l'explainer unifié (combine SHAP + LIME + Counterfactuals)
explainer = create_explainer(
    model,
    method='unified',
    methods=['shap', 'lime', 'counterfactual'],
    background_data=X_train.sample(100)
)

# 2. Expliquer une décision de rejet
rejected_applicant = X_test.iloc[0]
explanation = explainer.explain(rejected_applicant)

print("Pourquoi ce crédit a été rejeté:")
print(explanation.feature_importance)

# 3. Générer des recommandations actionnables
if hasattr(explanation, 'counterfactuals'):
    print("\nPour obtenir l'approbation, le client devrait:")
    for cf in explanation.counterfactuals[:3]:
        print(f"  - {cf}")

# 4. Vérifier la conformité GDPR (obligatoire en UE!)
gdpr = GDPRCompliance(model, model_metadata={
    'name': 'Modèle de Scoring Crédit',
    'purpose': 'Approbation de prêts',
    'legal_basis': 'legitimate_interest'
})

# Générer le rapport DPIA (requis par GDPR Article 35)
dpia_report = gdpr.generate_dpia()
dpia_report.export('gdpr_dpia_report.pdf')

print("\n✅ Rapport GDPR généré: gdpr_dpia_report.pdf")

# 5. Vérifier la conformité EU AI Act (HIGH RISK pour le crédit!)
ai_act = AIActCompliance(model, usage_intent='credit_scoring')
risk_category = ai_act.assess_risk_category()
print(f"\nCatégorie de risque AI Act: {risk_category}")

compliance_report = ai_act.generate_compliance_report()
compliance_report.export('eu_ai_act_report.pdf')

# 6. Évaluer la confiance et détecter le fairwashing
uq = UncertaintyQuantifier(model, explainer)
uncertainty = uq.quantify(X_test[:100])

print(f"\nIncertitude moyenne: {uncertainty.total_uncertainty.mean():.3f}")
print(f"Incertitude épistémique: {uncertainty.epistemic_uncertainty.mean():.3f}")
print(f"Incertitude aléatoire: {uncertainty.aleatoric_uncertainty.mean():.3f}")

# Détecter si l'explication cache des biais (UNIQUE à XPLIA!)
detector = FairwashingDetector(model, explainer)
fairwashing = detector.detect(X_test[:100], y_test[:100])

if fairwashing.detected:
    print(f"\n⚠️ ALERTE: Fairwashing détecté!")
    print(f"Types: {fairwashing.fairwashing_types}")
    print(f"Sévérité: {fairwashing.severity}")
else:
    print("\n✅ Aucun fairwashing détecté")

# 7. Générer un rapport complet HTML
from xplia.visualizations import ChartGenerator
chart_gen = ChartGenerator()

chart_gen.create_dashboard(
    explanation,
    uncertainty=uncertainty,
    fairwashing=fairwashing,
    output='credit_decision_report.html'
)

print("\n✅ Rapport complet généré: credit_decision_report.html")
```

### 2. Santé - Diagnostic Médical

```python
from xplia import create_explainer
from xplia.compliance import HIPAACompliance
from xplia.explainers.trust import UncertaintyQuantifier
from sklearn.ensemble import GradientBoostingClassifier

# Vos données médicales
X_train = pd.read_csv('medical_data_train.csv')
y_train = pd.read_csv('diagnosis_train.csv')
patient_data = pd.read_csv('patient_to_diagnose.csv')

# Modèle de diagnostic
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Explainer pour diagnostic médical
explainer = create_explainer(
    model,
    method='unified',
    methods=['shap', 'lime'],
    background_data=X_train.sample(50)
)

# Expliquer le diagnostic
explanation = explainer.explain(patient_data.iloc[0])

print("Facteurs influençant le diagnostic:")
for feature, importance in explanation.feature_importance.items():
    if importance > 0.05:  # Seulement les facteurs significatifs
        print(f"  - {feature}: {importance:.2%}")

# Quantifier l'incertitude (CRUCIAL en médecine!)
uq = UncertaintyQuantifier(model, explainer)
uncertainty = uq.quantify(patient_data.iloc[0:1])

print(f"\nConfiance du diagnostic: {1 - uncertainty.total_uncertainty[0]:.1%}")
print(f"Incertitude du modèle: {uncertainty.epistemic_uncertainty[0]:.3f}")
print(f"Variabilité inhérente: {uncertainty.aleatoric_uncertainty[0]:.3f}")

if uncertainty.total_uncertainty[0] > 0.3:
    print("\n⚠️ Incertitude élevée - Recommander un examen complémentaire")

# Conformité HIPAA
hipaa = HIPAACompliance(model)
audit_trail = hipaa.log_access(
    user_id='dr_smith_123',
    patient_id='patient_456',
    purpose='diagnostic_support',
    explanation=explanation
)

print(f"\n✅ Accès enregistré dans le journal HIPAA: {audit_trail.id}")
```

### 3. E-commerce - Recommandations Produits

```python
from xplia.explainers.recommender import CollaborativeFilteringExplainer, MatrixFactorizationExplainer
import numpy as np

# Matrice utilisateur-produit (ex: 1000 users x 500 produits)
user_item_matrix = np.load('user_item_ratings.npy')
user_features = np.load('user_features.npy')
item_features = np.load('item_features.npy')

# Créer l'explainer pour système de recommandation
cf_explainer = CollaborativeFilteringExplainer(
    user_item_matrix=user_item_matrix,
    similarity_metric='cosine'
)

# Expliquer pourquoi un produit est recommandé
user_id = 42
item_id = 123

explanation = cf_explainer.explain_recommendation(
    user_id=user_id,
    item_id=item_id
)

print(f"Pourquoi recommander le produit {item_id} à l'utilisateur {user_id}:")
print(f"\nUtilisateurs similaires: {explanation.similar_users[:5]}")
print(f"Produits similaires aimés: {explanation.similar_items_liked[:5]}")
print(f"Score de similarité: {explanation.similarity_score:.2f}")

# Factorisation matricielle pour comprendre les préférences latentes
mf_explainer = MatrixFactorizationExplainer(n_factors=20)
mf_explainer.fit(user_item_matrix)

latent_explanation = mf_explainer.explain_recommendation(user_id, item_id)

print(f"\nFacteurs latents importants:")
for factor_idx, importance in latent_explanation.factor_importance.items():
    print(f"  - Facteur {factor_idx}: {importance:.3f}")
```

### 4. Vision par Ordinateur - Classification d'Images

```python
from xplia.explainers.multimodal import CLIPExplainer
from xplia import create_explainer
from PIL import Image
import numpy as np

# Pour les modèles de vision traditionnels (CNN)
from torchvision.models import resnet50
import torch

model = resnet50(pretrained=True)
model.eval()

# Charger une image
image = Image.open('cat.jpg')
image_array = np.array(image)

# Expliquer avec Gradient-CAM
explainer = create_explainer(model, method='gradients', task='vision')
explanation = explainer.explain(image_array, layer_name='layer4')

# Visualiser les zones importantes
from xplia.visualizations import ChartGenerator
chart_gen = ChartGenerator()

chart_gen.create_heatmap(
    heatmap_data=explanation.attribution_map,
    original_image=image_array,
    title='Zones importantes pour la classification',
    output='gradcam_explanation.png'
)

print("✅ Heatmap d'explication générée: gradcam_explanation.png")

# Pour les modèles Vision-Language (CLIP, BLIP)
from xplia.explainers.multimodal import CLIPExplainer

clip_explainer = CLIPExplainer()
vl_explanation = clip_explainer.explain(
    image=image_array,
    text="a photo of a cat",
    method='attention'
)

print("\nAttention cross-modale (image-texte):")
print(f"Score de similarité: {vl_explanation.similarity_score:.3f}")
print(f"Régions importantes: {vl_explanation.important_regions[:5]}")
print(f"Tokens texte importants: {vl_explanation.important_text_tokens}")
```

### 5. NLP - Analyse de Sentiment / Classification de Texte

```python
from xplia import create_explainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Charger un modèle BERT pour analyse de sentiment
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Texte à expliquer
text = "This movie was absolutely terrible! The acting was horrible."

# Créer l'explainer pour transformers
explainer = create_explainer(
    model,
    method='attention',
    tokenizer=tokenizer,
    task='text_classification'
)

# Expliquer la prédiction
explanation = explainer.explain(text)

print("Analyse de sentiment:")
print(f"Prédiction: {explanation.prediction}")
print(f"Confiance: {explanation.confidence:.2%}")

print("\nImportance des mots:")
for token, importance in zip(explanation.tokens, explanation.token_importance):
    if importance > 0.05:
        print(f"  '{token}': {importance:.3f}")

# Visualiser l'attention
from xplia.visualizations import ChartGenerator
chart_gen = ChartGenerator()

chart_gen.create_text_heatmap(
    tokens=explanation.tokens,
    importances=explanation.token_importance,
    title='Importance des tokens pour le sentiment',
    output='text_explanation.html'
)
```

### 6. Séries Temporelles - Prévision / Détection d'Anomalies

```python
from xplia.explainers.timeseries import TemporalImportanceExplainer, AnomalyExplainer
import pandas as pd

# Vos données temporelles (ex: prix d'actions, température, ventes)
timeseries_data = pd.read_csv('stock_prices.csv', parse_dates=['date'], index_col='date')

# Modèle de prévision (LSTM, Prophet, etc.)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

# Créer des features temporelles
X = create_temporal_features(timeseries_data)  # lags, rolling means, etc.
y = timeseries_data['price'].shift(-1).dropna()

model.fit(X[:-1], y)

# Expliquer une prévision
ts_explainer = TemporalImportanceExplainer(model, window_size=30)
explanation = ts_explainer.explain_forecast(
    timeseries=timeseries_data[-30:],
    forecast_horizon=1
)

print("Importance temporelle:")
print(f"Lag le plus important: {explanation.most_important_lag}")
print(f"Saisonnalité détectée: {explanation.seasonality}")

# Détection d'anomalies avec explications
anomaly_explainer = AnomalyExplainer(model, threshold=2.5)
anomalies = anomaly_explainer.detect_and_explain(timeseries_data)

print(f"\n{len(anomalies.anomaly_indices)} anomalies détectées:")
for idx, reason in zip(anomalies.anomaly_indices[:5], anomalies.anomaly_reasons[:5]):
    print(f"  - Date {idx}: {reason}")
```

### 7. Reinforcement Learning - Explication de Politique

```python
from xplia.explainers.reinforcement import PolicyExplainer, QValueExplainer
import gym
import numpy as np

# Votre agent RL (ex: DQN, A3C, PPO)
env = gym.make('CartPole-v1')
policy_network = load_your_trained_policy()  # Votre politique entraînée

# Expliquer les décisions de l'agent
policy_explainer = PolicyExplainer(policy_network, env)

# État actuel
state = env.reset()

# Expliquer pourquoi l'agent choisit cette action
explanation = policy_explainer.explain_action(state)

print("Action choisie:", explanation.action)
print(f"Probabilité: {explanation.action_probability:.2%}")

print("\nInfluence des features de l'état:")
for feature, importance in explanation.state_feature_importance.items():
    print(f"  - {feature}: {importance:.3f}")

# Expliquer les Q-values
q_explainer = QValueExplainer(policy_network)
q_explanation = q_explainer.explain_q_values(state)

print("\nQ-values pour chaque action:")
for action, q_value in enumerate(q_explanation.q_values):
    print(f"  Action {action}: {q_value:.2f}")
```

### 8. Graph Neural Networks - Prédiction sur Graphes

```python
from xplia.explainers.graph import GNNExplainer
import torch
from torch_geometric.data import Data

# Votre GNN (ex: GCN, GAT, GraphSAGE)
gnn_model = load_your_gnn_model()

# Graphe à expliquer (ex: réseau social, molécule)
graph_data = Data(
    x=node_features,  # Features des noeuds
    edge_index=edge_index,  # Connexions
    y=node_labels
)

# Expliquer la prédiction pour un noeud
gnn_explainer = GNNExplainer(gnn_model)
explanation = gnn_explainer.explain_node(
    graph=graph_data,
    node_idx=42
)

print(f"Sous-graphe important:")
print(f"  - Noeuds: {explanation.important_nodes}")
print(f"  - Arêtes: {explanation.important_edges}")
print(f"  - Features: {explanation.important_features}")

# Visualiser le sous-graphe explicatif
from xplia.visualizations import GraphVisualizer
viz = GraphVisualizer()
viz.plot_explanation(
    graph=graph_data,
    node_idx=42,
    explanation=explanation,
    output='gnn_explanation.html'
)
```

---

## 🎯 Guide par Type de Modèle

### scikit-learn (Random Forest, SVM, etc.)

```python
from xplia import create_explainer
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# SHAP est optimal pour les arbres
explainer = create_explainer(model, method='shap', background_data=X_train.sample(100))
explanation = explainer.explain(X_test[:5])
```

### XGBoost / LightGBM / CatBoost

```python
from xplia import create_explainer
import xgboost as xgb

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# SHAP TreeExplainer est le plus rapide pour les boosting
explainer = create_explainer(model, method='shap', background_data=X_train[:100])
explanation = explainer.explain(X_test[:10])
```

### PyTorch (CNN, LSTM, Transformers)

```python
from xplia import create_explainer
import torch.nn as nn

class MyModel(nn.Module):
    # Votre architecture
    pass

model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Gradients pour les réseaux de neurones
explainer = create_explainer(
    model,
    method='gradients',
    input_shape=(3, 224, 224)  # Pour images
)

explanation = explainer.explain(test_image)
```

### TensorFlow / Keras

```python
from xplia import create_explainer
import tensorflow as tf

model = tf.keras.models.load_model('my_model.h5')

explainer = create_explainer(
    model,
    method='gradients',
    input_names=['input_1']
)

explanation = explainer.explain(X_test[:5])
```

---

## 🔥 Fonctionnalités Avancées

### 1. Adapter les Explications à Différents Publics

```python
from xplia.explainers.calibration import AudienceAdapter

adapter = AudienceAdapter()

# Explication technique pour data scientists
expert_exp = adapter.adapt(explanation, audience='expert')
print(expert_exp.detailed_metrics)

# Explication business pour executives
business_exp = adapter.adapt(explanation, audience='basic')
print(business_exp.summary)  # Langage simple, pas de jargon

# Explication pour le grand public
public_exp = adapter.adapt(explanation, audience='novice')
print(public_exp.simple_explanation)  # Très accessible
```

### 2. API REST pour Intégration

```python
from xplia.api import create_api_app
from fastapi import FastAPI

# Créer l'API
app = create_api_app(models={
    'credit_model': credit_model,
    'fraud_model': fraud_model
})

# Lancer le serveur
# uvicorn main:app --host 0.0.0.0 --port 8000

# Utiliser l'API depuis un client
import requests

response = requests.post('http://localhost:8000/explain', json={
    'model_name': 'credit_model',
    'instances': [[25, 50000, 3, 0.2]],
    'method': 'shap'
})

explanation = response.json()
print(explanation)
```

### 3. Intégration MLflow

```python
from xplia.integrations.mlflow import XPLIAMLflowLogger
import mlflow

with XPLIAMLflowLogger(experiment_name="credit_scoring") as logger:
    # Entraîner
    model.fit(X_train, y_train)

    # Logger automatiquement les explications
    explainer = create_explainer(model, method='shap')
    explanation = explainer.explain(X_test[:100])

    logger.log_model(model, 'credit_model')
    logger.log_explanation(explanation)
    logger.log_metrics(explanation.quality_metrics)
```

### 4. Weights & Biases Integration

```python
from xplia.integrations.wandb import XPLIAWandBContext
import wandb

with XPLIAWandBContext(project="fraud-detection") as logger:
    # Entraînement et logging automatique
    model.fit(X_train, y_train)

    explainer = create_explainer(model)
    explanation = explainer.explain(X_test)

    logger.log_explanation(explanation)
    logger.log_visualizations(explanation)
```

---

## 🐛 Résolution de Problèmes

### Problème: Installation échoue

```bash
# Solution 1: Mettre à jour pip
pip install --upgrade pip setuptools wheel

# Solution 2: Installer sans dépendances optionnelles
pip install xplia --no-deps
pip install numpy pandas scikit-learn matplotlib

# Solution 3: Utiliser conda
conda install -c conda-forge xplia
```

### Problème: Méthode SHAP très lente

```python
# Solution 1: Réduire les données de background
explainer = create_explainer(
    model,
    method='shap',
    background_data=X_train.sample(50)  # Au lieu de 100+
)

# Solution 2: Utiliser approximation pour deep learning
explainer = create_explainer(
    model,
    method='shap',
    approximation=True,
    n_samples=100
)

# Solution 3: Utiliser LIME à la place (plus rapide)
explainer = create_explainer(model, method='lime')
```

### Problème: Mémoire insuffisante

```python
from xplia import set_config

# Activer l'optimisation mémoire
set_config('memory_optimization', True)

# Traiter par batches
for batch in chunks(X_test, batch_size=10):
    explanation = explainer.explain(batch)
    # Traiter l'explication
    del explanation  # Libérer la mémoire
```

### Problème: Explication de mauvaise qualité

```python
# Solution 1: Augmenter les données de background
explainer = create_explainer(
    model,
    method='shap',
    background_data=X_train.sample(500)  # Plus de contexte
)

# Solution 2: Utiliser l'explainer unifié (consensus de plusieurs méthodes)
explainer = create_explainer(
    model,
    method='unified',
    methods=['shap', 'lime', 'gradients']
)

# Solution 3: Calibrer l'explainer
from xplia.explainers.calibration import ExplainerCalibrator

calibrator = ExplainerCalibrator()
calibrated_explainer = calibrator.calibrate(explainer, X_val, y_val)
```

---

## 📞 Support

- **Documentation**: https://xplia.readthedocs.io
- **GitHub Issues**: https://github.com/nicolasseverino/xplia/issues
- **Email**: contact@xplia.com

---

## 🎓 Prochaines Étapes

1. Parcourir les exemples dans `examples/`
2. Lire la documentation complète
3. Rejoindre la communauté
4. Contribuer au projet!
