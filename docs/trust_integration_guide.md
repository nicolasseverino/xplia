# Guide d'Intégration des Modules de Confiance XPLIA

## Introduction

Ce guide explique comment intégrer les modules avancés d'évaluation de confiance de XPLIA dans vos applications et rapports. Ces modules permettent d'enrichir les explications avec des métriques de fiabilité, de détecter le fairwashing potentiel, et de générer des rapports de confiance complets.

## Table des matières

1. [Prérequis](#prérequis)
2. [Intégration dans le Pipeline d'Explication](#intégration-dans-le-pipeline-dexplication)
3. [Intégration avec les Formatters](#intégration-avec-les-formatters)
4. [Personnalisation des Métriques](#personnalisation-des-métriques)
5. [Exemples Complets](#exemples-complets)
6. [Bonnes Pratiques](#bonnes-pratiques)
7. [Dépannage](#dépannage)

## Prérequis

Avant d'intégrer les modules de confiance, assurez-vous d'avoir installé XPLIA avec toutes les dépendances optionnelles :

```bash
pip install xplia[all]
```

Pour une installation minimale avec uniquement les modules de confiance :

```bash
pip install xplia[trust]
```

## Intégration dans le Pipeline d'Explication

### Étape 1 : Importer les modules nécessaires

```python
from xplia.explainers import ShapExplainer, UncertaintyQuantifier, FairwashingDetector, ConfidenceReport
from xplia.core.model_adapters import SklearnAdapter  # Ou un autre adaptateur selon votre modèle
```

### Étape 2 : Créer l'explication de base

```python
# Créer l'adaptateur de modèle
model_adapter = SklearnAdapter(model)

# Créer l'explainer
explainer = ShapExplainer()

# Générer l'explication
explanation = explainer.explain(model_adapter, instance, background_data=X_train)
```

### Étape 3 : Quantifier l'incertitude

```python
# Créer le quantificateur d'incertitude
uncertainty_quantifier = UncertaintyQuantifier(
    n_bootstrap_samples=50,
    confidence_level=0.95,
    methods=["bootstrap", "ensemble", "sensitivity"]
)

# Quantifier l'incertitude
uncertainty_metrics = uncertainty_quantifier.quantify_uncertainty(
    explanation=explanation,
    explainer=explainer,
    X=X_test.sample(20)  # Échantillon pour l'estimation
)
```

### Étape 4 : Détecter le fairwashing

```python
# Créer le détecteur de fairwashing
fairwashing_detector = FairwashingDetector(
    sensitive_features=['gender', 'race'],  # Features sensibles
    detection_threshold=0.7,
    methods=["consistency", "sensitivity", "counterfactual"]
)

# Détecter le fairwashing
fairwashing_audit = fairwashing_detector.detect_fairwashing(
    explanation=explanation,
    X=X_test  # Données pour la détection
)
```

### Étape 5 : Générer un rapport de confiance

```python
# Créer le générateur de rapports de confiance
confidence_reporter = ConfidenceReport(
    uncertainty_weight=0.4,
    fairwashing_weight=0.3,
    consistency_weight=0.2,
    robustness_weight=0.1
)

# Générer le rapport de confiance
confidence_report = confidence_reporter.generate_report(
    explanation=explanation,
    uncertainty_metrics=uncertainty_metrics,
    fairwashing_audit=fairwashing_audit
)

# Enrichir l'explication avec le rapport de confiance
explanation = confidence_reporter.apply_to_explanation(explanation, confidence_report)
```

## Intégration avec les Formatters

XPLIA fournit des formatters spécialisés pour intégrer les métriques de confiance dans les rapports HTML et PDF.

### Formatter HTML avec Métriques de Confiance

```python
from xplia.compliance.formatters.html_trust_formatter import TrustHTMLReportGenerator
from xplia.compliance.report_base import ReportConfig, ReportContent

# Configuration du rapport
config = ReportConfig(
    template_name="standard",
    language="fr",
    include_verification_qr=True,
    verification_url="https://xplia.ai/verify"
)

# Création du générateur de rapports HTML avec métriques de confiance
html_generator = TrustHTMLReportGenerator(config)

# Création du contenu du rapport
content = ReportContent(
    title="Rapport d'Explication avec Métriques de Confiance",
    description="Ce rapport présente une explication avec évaluation de confiance.",
    explanations=[explanation],  # L'explication enrichie avec le rapport de confiance
    metadata={
        "model_type": "RandomForestClassifier",
        "date_created": "2025-06-13T08:30:00+02:00",
        "authors": ["XPLIA System"]
    }
)

# Génération du rapport
html_generator.generate(content, "rapport_confiance.html")
```

### Formatter PDF avec Métriques de Confiance

```python
from xplia.compliance.formatters.pdf_trust_formatter import TrustPDFReportGenerator

# Création du générateur de rapports PDF avec métriques de confiance
pdf_generator = TrustPDFReportGenerator(config)

# Génération du rapport
pdf_generator.generate(content, "rapport_confiance.pdf")
```

## Personnalisation des Métriques

### Personnalisation de la Quantification d'Incertitude

```python
# Personnalisation des méthodes d'estimation
uncertainty_quantifier = UncertaintyQuantifier(
    n_bootstrap_samples=100,
    confidence_level=0.99,
    methods=["bootstrap", "ensemble", "sensitivity", "variance", "bayesian"]
)

# Personnalisation des types d'incertitude à estimer
uncertainty_metrics = uncertainty_quantifier.quantify_uncertainty(
    explanation=explanation,
    explainer=explainer,
    X=X_test.sample(30),
    uncertainty_types=[UncertaintyType.ALEATORIC, UncertaintyType.EPISTEMIC],
    feature_subset=['feature1', 'feature2']  # Limiter à certaines features
)
```

### Personnalisation de la Détection de Fairwashing

```python
# Personnalisation des méthodes de détection
fairwashing_detector = FairwashingDetector(
    sensitive_features=['gender', 'race', 'age'],
    detection_threshold=0.6,
    methods=["consistency", "sensitivity", "counterfactual", "statistical", "adversarial"]
)

# Personnalisation des types de fairwashing à détecter
fairwashing_audit = fairwashing_detector.detect_fairwashing(
    explanation=explanation,
    X=X_test,
    fairwashing_types=[FairwashingType.FEATURE_MASKING, FairwashingType.IMPORTANCE_SHIFT],
    reference_explanation=reference_explanation  # Explication de référence pour comparaison
)
```

### Personnalisation du Rapport de Confiance

```python
# Personnalisation des poids des dimensions
confidence_reporter = ConfidenceReport(
    uncertainty_weight=0.5,    # Plus d'importance à l'incertitude
    fairwashing_weight=0.3,
    consistency_weight=0.1,
    robustness_weight=0.1
)

# Ajout de métriques personnalisées
additional_metrics = {
    "custom_metric_1": 0.85,
    "custom_metric_2": 0.72
}

confidence_report = confidence_reporter.generate_report(
    explanation=explanation,
    uncertainty_metrics=uncertainty_metrics,
    fairwashing_audit=fairwashing_audit,
    additional_metrics=additional_metrics
)
```

## Exemples Complets

XPLIA fournit plusieurs exemples complets d'intégration des modules de confiance :

1. **trust_metrics_simple_demo.py** : Démonstration simple des modules de confiance
2. **trust_formatter_demo.py** : Démonstration de l'intégration avec les formatters
3. **trust_pipeline_demo.py** : Démonstration du pipeline complet d'évaluation de confiance

Pour exécuter ces exemples :

```bash
python examples/trust_metrics_simple_demo.py
python examples/trust_formatter_demo.py
python examples/trust_pipeline_demo.py
```

## Bonnes Pratiques

1. **Taille d'échantillon** : Utilisez un échantillon suffisamment grand pour l'estimation d'incertitude (au moins 20-30 instances).

2. **Features sensibles** : Identifiez correctement les features sensibles pour la détection de fairwashing.

3. **Équilibrage des poids** : Ajustez les poids des dimensions de confiance selon votre cas d'utilisation.

4. **Visualisation** : Utilisez les visualisations intégrées pour mieux comprendre les métriques de confiance.

5. **Interprétation contextuelle** : Interprétez les métriques de confiance dans le contexte de votre domaine d'application.

## Dépannage

### Problèmes courants

1. **Erreurs d'importation** : Assurez-vous d'avoir installé toutes les dépendances optionnelles.

   ```bash
   pip install xplia[all]
   ```

2. **Erreurs de mémoire** : Réduisez la taille des échantillons ou le nombre d'échantillons bootstrap.

3. **Lenteur de calcul** : Utilisez des méthodes d'estimation plus légères ou réduisez la complexité.

4. **Résultats incohérents** : Augmentez la taille des échantillons ou utilisez des méthodes plus robustes.

### Support

Pour plus d'aide, consultez la documentation complète ou contactez le support XPLIA :

- Documentation : [https://xplia.ai/docs](https://xplia.ai/docs)
- Support : [support@xplia.ai](mailto:support@xplia.ai)
