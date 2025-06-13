# Guide d'utilisation des modules d'évaluation experte et de métriques de confiance

## Introduction

Ce guide explique comment utiliser les modules d'évaluation experte et de métriques de confiance de XPLIA pour évaluer la qualité des explications et la fiabilité des modèles d'IA.

Les modules d'évaluation experte permettent de :
- Quantifier la qualité des explications générées
- Mesurer l'incertitude des prédictions
- Détecter les cas de fairwashing (blanchiment d'équité)
- Générer des rapports de confiance complets
- Visualiser les résultats d'évaluation

## Prérequis

Pour utiliser les modules d'évaluation experte, vous devez avoir installé XPLIA avec toutes ses dépendances :

```bash
pip install -e .
```

## Modules disponibles

### 1. Évaluation de la qualité des explications

Le module `ExplanationQualityEvaluator` permet d'évaluer la qualité des explications selon plusieurs critères :

- Fidélité au modèle
- Complétude de l'explication
- Cohérence entre instances
- Robustesse de l'explication
- Simplicité et clarté
- Équité de l'explication
- Transparence des méthodes
- Actionabilité
- Performance de génération
- Quantification de l'incertitude

### 2. Évaluation des métriques de confiance

Le module `TrustExpertEvaluator` permet d'évaluer les métriques de confiance selon plusieurs critères :

- Précision de l'incertitude
- Détection de fairwashing
- Cohérence des métriques
- Robustesse aux perturbations
- Complétude des métriques
- Transparence des calculs
- Interprétabilité des scores
- Utilité des recommandations
- Performance de calcul
- Calibration des intervalles

### 3. Intégration des évaluations

Le module `ExpertEvaluationIntegrator` permet d'intégrer les évaluations de qualité et de confiance dans un pipeline complet.

## Exemples d'utilisation

### Évaluation de la qualité d'une explication

```python
from xplia.explainers.expert_evaluator import ExplanationQualityEvaluator
from xplia.explainers import lime_explainer

# Création de l'explainer et génération de l'explication
explainer = lime_explainer.LimeExplainer()
explanation = explainer.explain(
    model_adapter=model_adapter,
    instance=instance,
    background_data=X_train
)

# Évaluation de la qualité de l'explication
evaluator = ExplanationQualityEvaluator()
quality_review = evaluator.evaluate_explanation(
    explanation=explanation,
    model_adapter=model_adapter,
    instance=instance,
    background_data=X_train
)

# Affichage des résultats
print(f"Score global : {quality_review.global_score:.2f}/10")
print("Points forts :")
for strength in quality_review.strengths:
    print(f"- {strength}")
print("Points faibles :")
for weakness in quality_review.weaknesses:
    print(f"- {weakness}")
print("Recommandations :")
for recommendation in quality_review.recommendations:
    print(f"- {recommendation}")
```

### Évaluation des métriques de confiance

```python
from xplia.trust.uncertainty import UncertaintyEstimator
from xplia.trust.fairwashing import FairwashingAuditor
from xplia.trust.confidence import ConfidenceReporter
from xplia.compliance.expert_review.trust_expert_evaluator import TrustExpertEvaluator

# Calcul des métriques d'incertitude
uncertainty_estimator = UncertaintyEstimator()
uncertainty_metrics = uncertainty_estimator.estimate(
    model_adapter=model_adapter,
    instance=instance,
    explanation=explanation,
    background_data=X_train
)

# Audit de fairwashing
fairwashing_auditor = FairwashingAuditor()
fairwashing_audit = fairwashing_auditor.audit(
    model_adapter=model_adapter,
    instance=instance,
    explanation=explanation,
    background_data=X_train,
    sensitive_features=sensitive_features
)

# Génération du rapport de confiance
confidence_reporter = ConfidenceReporter()
confidence_report = confidence_reporter.generate_report(
    explanation=explanation,
    uncertainty_metrics=uncertainty_metrics,
    fairwashing_audit=fairwashing_audit
)

# Évaluation des métriques de confiance
trust_evaluator = TrustExpertEvaluator()
trust_review = trust_evaluator.evaluate_trust_metrics(
    uncertainty_metrics=uncertainty_metrics,
    fairwashing_audit=fairwashing_audit,
    confidence_report=confidence_report,
    explanation=explanation
)

# Affichage des résultats
print(f"Score global de confiance : {trust_review.global_score:.2f}/10")
print("Points forts :")
for strength in trust_review.strengths:
    print(f"- {strength}")
print("Points faibles :")
for weakness in trust_review.weaknesses:
    print(f"- {weakness}")
print("Recommandations :")
for recommendation in trust_review.recommendations:
    print(f"- {recommendation}")
```

### Utilisation de l'intégrateur d'évaluation

```python
from xplia.compliance.expert_review.integration import ExpertEvaluationIntegrator
from xplia.explainers import lime_explainer

# Création de l'explainer
explainer = lime_explainer.LimeExplainer()

# Création de l'intégrateur
integrator = ExpertEvaluationIntegrator()

# Évaluation complète du pipeline d'explication
results = integrator.evaluate_explanation_pipeline(
    explainer=explainer,
    model_adapter=model_adapter,
    instance=instance,
    background_data=X_train,
    sensitive_features=sensitive_features
)

# Génération d'un rapport complet
report = integrator.generate_comprehensive_report(
    evaluation_results=results,
    include_visualizations=True
)

# Affichage des résultats
print(f"Score global : {report['summary']['overall_score']:.2f}/10")
print(f"Score de qualité : {report['summary']['explanation_quality_score']:.2f}/10")
print(f"Score de confiance : {report['summary']['trust_score']:.2f}/10")
print("Recommandations :")
for recommendation in report["recommendations"]:
    print(f"- {recommendation}")
```

### Utilisation de la fonction d'évaluation rapide

```python
from xplia.compliance.expert_review.integration import quick_evaluate
from xplia.explainers import lime_explainer

# Création de l'explainer
explainer = lime_explainer.LimeExplainer()

# Évaluation rapide
report = quick_evaluate(
    explainer=explainer,
    model_adapter=model_adapter,
    instance=instance,
    background_data=X_train,
    sensitive_features=sensitive_features
)

# Affichage des résultats
print(f"Score global : {report['summary']['overall_score']:.2f}/10")
```

## Visualisation des résultats

Les résultats d'évaluation peuvent être visualisés à l'aide du module `ChartGenerator` :

```python
from xplia.visualizations import ChartGenerator

# Création du générateur de graphiques
chart_generator = ChartGenerator()

# Visualisation du score global
gauge_chart = chart_generator.gauge_chart(
    value=report["summary"]["overall_score"],
    min_value=0.0,
    max_value=10.0,
    title="Score global",
    thresholds=[3.0, 5.0, 7.0, 9.0]
)

# Affichage du graphique
# Pour HTML
with open("gauge_chart.html", "w") as f:
    f.write(gauge_chart)

# Pour les notebooks Jupyter
from IPython.display import HTML
HTML(gauge_chart)
```

## Démonstration interactive

Pour une démonstration interactive des métriques de confiance, vous pouvez utiliser le script `interactive_trust_demo.py` :

```bash
python examples/interactive_trust_demo.py
```

Ce script permet de :
- Charger différents jeux de données
- Entraîner différents modèles
- Générer des explications
- Calculer des métriques de confiance
- Évaluer la qualité des explications et des métriques de confiance
- Visualiser les résultats

## Intégration avec les rapports

Les évaluations expertes peuvent être intégrées dans les rapports HTML et PDF générés par XPLIA :

```python
from xplia.formatters.html_formatter import HTMLReportGenerator
from xplia.compliance.expert_review.integration import ExpertEvaluationIntegrator

# Évaluation du pipeline d'explication
integrator = ExpertEvaluationIntegrator()
results = integrator.evaluate_explanation_pipeline(
    explainer=explainer,
    model_adapter=model_adapter,
    instance=instance,
    background_data=X_train,
    sensitive_features=sensitive_features
)

# Génération du rapport HTML
html_generator = HTMLReportGenerator()
html_report = html_generator.generate(
    explanation=results["explanation"],
    model_adapter=model_adapter,
    instance=instance,
    additional_sections={
        "Évaluation experte": {
            "content": results["trust_evaluation"],
            "type": "expert_evaluation"
        }
    }
)

# Sauvegarde du rapport HTML
with open("report.html", "w") as f:
    f.write(html_report)
```

## Conclusion

Les modules d'évaluation experte de XPLIA permettent d'évaluer de manière objective la qualité des explications et la fiabilité des modèles d'IA. Ils fournissent des métriques quantitatives et des recommandations pour améliorer la qualité des explications et la confiance dans les modèles.

Pour plus d'informations, consultez la documentation API complète des modules d'évaluation experte.
