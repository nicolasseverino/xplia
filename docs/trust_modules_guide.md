# Guide des Modules d'Évaluation de Confiance XPLIA

## Introduction

Les modules d'évaluation de confiance de XPLIA fournissent des outils avancés pour évaluer la fiabilité des explications générées par les différents explainers. Ces modules permettent de quantifier l'incertitude, de détecter le fairwashing potentiel, et de générer des rapports de confiance complets.

## Table des matières

1. [Quantification d'Incertitude](#quantification-dincertitude)
2. [Détection de Fairwashing](#détection-de-fairwashing)
3. [Rapports de Confiance](#rapports-de-confiance)
4. [Intégration dans le Pipeline d'Explication](#intégration-dans-le-pipeline-dexplication)
5. [Exemples d'Utilisation](#exemples-dutilisation)
6. [API de Référence](#api-de-référence)

## Quantification d'Incertitude

Le module de quantification d'incertitude permet d'évaluer la fiabilité des explications en utilisant différentes méthodes d'estimation d'incertitude.

### Types d'Incertitude

- **Aléatoire (Aleatoric)**: Incertitude inhérente aux données
- **Épistémique (Epistemic)**: Incertitude due aux limites du modèle
- **Structurelle (Structural)**: Incertitude due à la structure du modèle
- **Approximation**: Incertitude due à l'approximation de l'explication
- **Échantillonnage (Sampling)**: Incertitude due à l'échantillonnage
- **Feature**: Incertitude sur les attributions de features

### Méthodes d'Estimation

- **Bootstrap**: Estimation par rééchantillonnage des données
- **Ensemble**: Utilisation de plusieurs explainers différents
- **Analyse de Sensibilité**: Perturbation des données d'entrée
- **Estimation de Variance**: Analyse de la variance des attributions
- **Approche Bayésienne**: Utilisation de méthodes bayésiennes

### Exemple d'Utilisation

```python
from xplia.explainers import UncertaintyQuantifier, ShapExplainer
from xplia.core.model_adapters import SklearnAdapter

# Créer un explainer et un quantificateur d'incertitude
explainer = ShapExplainer()
uncertainty_quantifier = UncertaintyQuantifier(n_bootstrap_samples=50)

# Générer une explication
explanation = explainer.explain(model_adapter, instance, background_data=X_train)

# Quantifier l'incertitude
uncertainty_metrics = uncertainty_quantifier.quantify_uncertainty(
    explanation, explainer, X_test.sample(20)
)

# Afficher les métriques d'incertitude
print(f"Incertitude globale: {uncertainty_metrics.global_uncertainty}")
print(f"Incertitude aléatoire: {uncertainty_metrics.aleatoric_uncertainty}")
print(f"Incertitude épistémique: {uncertainty_metrics.epistemic_uncertainty}")
```

## Détection de Fairwashing

Le module de détection de fairwashing permet d'identifier les explications potentiellement trompeuses ou manipulées pour masquer des biais.

### Types de Fairwashing

- **Masquage de Features (Feature Masking)**: Masquage de features sensibles
- **Déplacement d'Importance (Importance Shift)**: Déplacement d'importance entre features
- **Dissimulation de Biais (Bias Hiding)**: Dissimulation de biais
- **Sélection Biaisée (Cherry Picking)**: Sélection biaisée d'exemples
- **Manipulation de Seuils (Threshold Manipulation)**: Manipulation des seuils

### Méthodes de Détection

- **Cohérence (Consistency)**: Comparaison entre deux explications
- **Sensibilité (Sensitivity)**: Analyse de la sensibilité aux features sensibles
- **Contrefactuels (Counterfactual)**: Vérification avec des exemples contrefactuels
- **Analyse Statistique**: Analyse statistique des attributions
- **Tests Adversariaux (Adversarial)**: Tests avec des exemples adversariaux

### Exemple d'Utilisation

```python
from xplia.explainers import FairwashingDetector, ShapExplainer
from xplia.core.model_adapters import SklearnAdapter

# Définir les features sensibles
sensitive_features = ['age', 'gender', 'race']

# Créer un explainer et un détecteur de fairwashing
explainer = ShapExplainer()
fairwashing_detector = FairwashingDetector(sensitive_features=sensitive_features)

# Générer une explication
explanation = explainer.explain(model_adapter, instance)

# Détecter le fairwashing
fairwashing_audit = fairwashing_detector.detect_fairwashing(
    explanation, X=X_test.sample(20)
)

# Afficher les résultats de l'audit
print(f"Score de fairwashing: {fairwashing_audit.fairwashing_score}")
print(f"Types détectés: {[t.value for t in fairwashing_audit.detected_types]}")
```

## Rapports de Confiance

Le module de rapports de confiance permet de générer des rapports complets intégrant les métriques d'incertitude et les résultats de détection de fairwashing.

### Niveaux de Confiance

- **Très Faible (Very Low)**: Confiance très faible
- **Faible (Low)**: Confiance faible
- **Modérée (Moderate)**: Confiance modérée
- **Élevée (High)**: Confiance élevée
- **Très Élevée (Very High)**: Confiance très élevée

### Dimensions de Confiance

- **Incertitude (Uncertainty)**: Confiance liée à l'incertitude
- **Fairwashing**: Confiance liée au fairwashing
- **Cohérence (Consistency)**: Confiance liée à la cohérence
- **Robustesse (Robustness)**: Confiance liée à la robustesse

### Exemple d'Utilisation

```python
from xplia.explainers import ConfidenceReport, UncertaintyQuantifier, FairwashingDetector
from xplia.explainers import ShapExplainer
from xplia.core.model_adapters import SklearnAdapter

# Créer les évaluateurs de confiance
uncertainty_quantifier = UncertaintyQuantifier()
fairwashing_detector = FairwashingDetector()
confidence_reporter = ConfidenceReport()

# Générer une explication
explainer = ShapExplainer()
explanation = explainer.explain(model_adapter, instance)

# Évaluer l'incertitude et le fairwashing
uncertainty_metrics = uncertainty_quantifier.quantify_uncertainty(
    explanation, explainer, X_test.sample(20)
)
fairwashing_audit = fairwashing_detector.detect_fairwashing(
    explanation, X=X_test.sample(20)
)

# Générer un rapport de confiance
confidence_report = confidence_reporter.generate_report(
    explanation, uncertainty_metrics, fairwashing_audit
)

# Afficher le rapport
print(f"Score de confiance global: {confidence_report['trust_score']['global_trust']}")
print(f"Niveau de confiance: {confidence_report['trust_score']['trust_level']}")
print(f"Résumé: {confidence_report['summary']}")
```

## Intégration dans le Pipeline d'Explication

Les modules d'évaluation de confiance peuvent être intégrés dans le pipeline d'explication de XPLIA pour enrichir les explications avec des métriques de fiabilité.

### Enrichissement des Explications

```python
from xplia.explainers import ShapExplainer, UncertaintyQuantifier, FairwashingDetector, ConfidenceReport
from xplia.core.model_adapters import SklearnAdapter

# Créer les évaluateurs de confiance
uncertainty_quantifier = UncertaintyQuantifier()
fairwashing_detector = FairwashingDetector()
confidence_reporter = ConfidenceReport()

# Générer une explication
explainer = ShapExplainer()
explanation = explainer.explain(model_adapter, instance)

# Évaluer l'incertitude et le fairwashing
uncertainty_metrics = uncertainty_quantifier.quantify_uncertainty(
    explanation, explainer, X_test.sample(20)
)
fairwashing_audit = fairwashing_detector.detect_fairwashing(
    explanation, X=X_test.sample(20)
)

# Générer un rapport de confiance
confidence_report = confidence_reporter.generate_report(
    explanation, uncertainty_metrics, fairwashing_audit
)

# Enrichir l'explication avec le rapport de confiance
explanation = confidence_reporter.apply_to_explanation(explanation, confidence_report)
```

### Intégration avec les Formatters

```python
from xplia.explainers import ShapExplainer, UncertaintyQuantifier, FairwashingDetector, ConfidenceReport
from xplia.formatters.html import HTMLReportGenerator
from xplia.core.model_adapters import SklearnAdapter

# Générer une explication avec évaluation de confiance
explainer = ShapExplainer()
explanation = explainer.explain(model_adapter, instance)

uncertainty_quantifier = UncertaintyQuantifier()
uncertainty_metrics = uncertainty_quantifier.quantify_uncertainty(
    explanation, explainer, X_test.sample(20)
)

fairwashing_detector = FairwashingDetector()
fairwashing_audit = fairwashing_detector.detect_fairwashing(
    explanation, X=X_test.sample(20)
)

confidence_reporter = ConfidenceReport()
confidence_report = confidence_reporter.generate_report(
    explanation, uncertainty_metrics, fairwashing_audit
)
explanation = confidence_reporter.apply_to_explanation(explanation, confidence_report)

# Générer un rapport HTML avec les métriques de confiance
html_generator = HTMLReportGenerator()
html_report = html_generator.generate_report(
    explanation=explanation,
    model_adapter=model_adapter,
    title="Rapport avec Évaluation de Confiance",
    include_confidence=True  # Inclure les métriques de confiance
)
```

## Exemples d'Utilisation

Voir le script de démonstration complet dans `examples/trust_pipeline_demo.py`.

## API de Référence

### Module `uncertainty`

#### Classe `UncertaintyQuantifier`

- `__init__(n_bootstrap_samples=100, confidence_level=0.95, methods=None)`: Initialise le quantificateur d'incertitude
- `quantify_uncertainty(explanation, explainer=None, X=None, y=None, **kwargs)`: Quantifie l'incertitude d'une explication
- `apply_to_explanation(explanation, metrics)`: Applique les métriques d'incertitude à une explication

#### Classe `UncertaintyMetrics`

- `global_uncertainty`: Score global d'incertitude
- `aleatoric_uncertainty`: Score d'incertitude aléatoire
- `epistemic_uncertainty`: Score d'incertitude épistémique
- `structural_uncertainty`: Score d'incertitude structurelle
- `approximation_uncertainty`: Score d'incertitude d'approximation
- `sampling_uncertainty`: Score d'incertitude d'échantillonnage
- `feature_uncertainties`: Incertitude par feature
- `confidence_intervals`: Intervalles de confiance
- `to_dict()`: Convertit les métriques en dictionnaire
- `from_dict(data)`: Crée des métriques à partir d'un dictionnaire

### Module `fairwashing`

#### Classe `FairwashingDetector`

- `__init__(sensitive_features=None, detection_threshold=0.7, methods=None)`: Initialise le détecteur de fairwashing
- `detect_fairwashing(explanation, reference_explanation=None, model=None, X=None, sensitive_groups=None, **kwargs)`: Détecte les signes de fairwashing dans une explication
- `apply_to_explanation(explanation, audit)`: Applique les résultats d'audit à une explication

#### Classe `FairwashingAudit`

- `fairwashing_score`: Score global de fairwashing
- `detected_types`: Types de fairwashing détectés
- `type_scores`: Scores par type de fairwashing
- `feature_manipulation_scores`: Scores de manipulation par feature
- `anomalies`: Anomalies détectées
- `to_dict()`: Convertit l'audit en dictionnaire
- `from_dict(data)`: Crée un audit à partir d'un dictionnaire

### Module `confidence_report`

#### Classe `ConfidenceReport`

- `__init__(uncertainty_weight=0.4, fairwashing_weight=0.3, consistency_weight=0.2, robustness_weight=0.1)`: Initialise le générateur de rapports
- `generate_report(explanation, uncertainty_metrics=None, fairwashing_audit=None, additional_metrics=None)`: Génère un rapport de confiance complet
- `calculate_trust_score(explanation, uncertainty_metrics=None, fairwashing_audit=None, additional_metrics=None)`: Calcule un score de confiance global
- `apply_to_explanation(explanation, report)`: Applique le rapport de confiance à une explication

#### Classe `TrustScore`

- `global_trust`: Score global de confiance
- `trust_level`: Niveau de confiance
- `uncertainty_trust`: Score de confiance lié à l'incertitude
- `fairwashing_trust`: Score de confiance lié au fairwashing
- `consistency_trust`: Score de confiance lié à la cohérence
- `robustness_trust`: Score de confiance lié à la robustesse
- `trust_factors`: Facteurs influençant le score
- `to_dict()`: Convertit le score en dictionnaire
- `from_dict(data)`: Crée un score à partir d'un dictionnaire
