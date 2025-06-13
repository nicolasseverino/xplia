# Documentation API des Modules de Confiance XPLIA

## Vue d'ensemble

Les modules de confiance XPLIA fournissent des outils avancés pour évaluer la fiabilité des explications d'IA. Cette documentation technique détaille les API disponibles pour les développeurs qui souhaitent intégrer ces fonctionnalités dans leurs applications.

## Modules principaux

### UncertaintyQuantifier

Module pour quantifier l'incertitude dans les explications d'IA.

#### Classes

##### `UncertaintyQuantifier`

```python
class UncertaintyQuantifier:
    def __init__(
        self, 
        n_bootstrap_samples: int = 50, 
        confidence_level: float = 0.95,
        methods: List[str] = ["bootstrap", "ensemble", "sensitivity"]
    )
```

**Paramètres :**
- `n_bootstrap_samples` : Nombre d'échantillons bootstrap pour l'estimation d'incertitude
- `confidence_level` : Niveau de confiance pour les intervalles (0-1)
- `methods` : Méthodes d'estimation d'incertitude à utiliser

**Méthodes :**

```python
def quantify_uncertainty(
    self,
    explanation: Dict[str, Any],
    explainer: Any,
    X: pd.DataFrame,
    uncertainty_types: List[UncertaintyType] = None,
    feature_subset: List[str] = None
) -> UncertaintyMetrics
```

**Paramètres :**
- `explanation` : Explication à évaluer
- `explainer` : Explainer utilisé pour générer l'explication
- `X` : Données pour l'estimation d'incertitude
- `uncertainty_types` : Types d'incertitude à estimer (optionnel)
- `feature_subset` : Sous-ensemble de features à évaluer (optionnel)

**Retourne :**
- `UncertaintyMetrics` : Métriques d'incertitude calculées

##### `UncertaintyMetrics`

```python
class UncertaintyMetrics:
    global_uncertainty: float
    aleatoric_uncertainty: float
    epistemic_uncertainty: float
    structural_uncertainty: float
    approximation_uncertainty: float
    feature_uncertainties: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
```

**Attributs :**
- `global_uncertainty` : Score global d'incertitude (0-1)
- `aleatoric_uncertainty` : Incertitude aléatoire (0-1)
- `epistemic_uncertainty` : Incertitude épistémique (0-1)
- `structural_uncertainty` : Incertitude structurelle (0-1)
- `approximation_uncertainty` : Incertitude d'approximation (0-1)
- `feature_uncertainties` : Scores d'incertitude par feature
- `confidence_intervals` : Intervalles de confiance par feature

##### `UncertaintyType`

```python
class UncertaintyType(Enum):
    ALEATORIC = "aleatoric"
    EPISTEMIC = "epistemic"
    STRUCTURAL = "structural"
    APPROXIMATION = "approximation"
```

**Valeurs :**
- `ALEATORIC` : Incertitude inhérente aux données
- `EPISTEMIC` : Incertitude due au manque de connaissances
- `STRUCTURAL` : Incertitude due à la structure du modèle
- `APPROXIMATION` : Incertitude due à l'approximation de l'explication

### FairwashingDetector

Module pour détecter le fairwashing dans les explications d'IA.

#### Classes

##### `FairwashingDetector`

```python
class FairwashingDetector:
    def __init__(
        self,
        sensitive_features: List[str],
        detection_threshold: float = 0.7,
        methods: List[str] = ["consistency", "sensitivity", "counterfactual"]
    )
```

**Paramètres :**
- `sensitive_features` : Liste des features sensibles à surveiller
- `detection_threshold` : Seuil de détection (0-1)
- `methods` : Méthodes de détection à utiliser

**Méthodes :**

```python
def detect_fairwashing(
    self,
    explanation: Dict[str, Any],
    X: pd.DataFrame,
    fairwashing_types: List[FairwashingType] = None,
    reference_explanation: Dict[str, Any] = None
) -> FairwashingAudit
```

**Paramètres :**
- `explanation` : Explication à évaluer
- `X` : Données pour la détection
- `fairwashing_types` : Types de fairwashing à détecter (optionnel)
- `reference_explanation` : Explication de référence pour comparaison (optionnel)

**Retourne :**
- `FairwashingAudit` : Résultats de l'audit de fairwashing

##### `FairwashingAudit`

```python
class FairwashingAudit:
    fairwashing_score: float
    detected_types: List[FairwashingType]
    type_scores: Dict[str, float]
    feature_manipulation_scores: Dict[str, float]
    counterfactual_analysis: Dict[str, Any]
```

**Attributs :**
- `fairwashing_score` : Score global de fairwashing (0-1)
- `detected_types` : Types de fairwashing détectés
- `type_scores` : Scores par type de fairwashing
- `feature_manipulation_scores` : Scores de manipulation par feature
- `counterfactual_analysis` : Résultats de l'analyse contrefactuelle

##### `FairwashingType`

```python
class FairwashingType(Enum):
    FEATURE_MASKING = "feature_masking"
    IMPORTANCE_SHIFT = "importance_shift"
    SELECTIVE_EMPHASIS = "selective_emphasis"
    COUNTERFACTUAL_MANIPULATION = "counterfactual_manipulation"
    THRESHOLD_MANIPULATION = "threshold_manipulation"
```

**Valeurs :**
- `FEATURE_MASKING` : Masquage de features sensibles
- `IMPORTANCE_SHIFT` : Modification artificielle de l'importance des features
- `SELECTIVE_EMPHASIS` : Mise en avant sélective de certaines features
- `COUNTERFACTUAL_MANIPULATION` : Manipulation des explications contrefactuelles
- `THRESHOLD_MANIPULATION` : Manipulation des seuils d'importance

### ConfidenceReport

Module pour générer des rapports de confiance complets.

#### Classes

##### `ConfidenceReport`

```python
class ConfidenceReport:
    def __init__(
        self,
        uncertainty_weight: float = 0.4,
        fairwashing_weight: float = 0.3,
        consistency_weight: float = 0.2,
        robustness_weight: float = 0.1
    )
```

**Paramètres :**
- `uncertainty_weight` : Poids de l'incertitude dans le score global
- `fairwashing_weight` : Poids du fairwashing dans le score global
- `consistency_weight` : Poids de la cohérence dans le score global
- `robustness_weight` : Poids de la robustesse dans le score global

**Méthodes :**

```python
def generate_report(
    self,
    explanation: Dict[str, Any],
    uncertainty_metrics: UncertaintyMetrics,
    fairwashing_audit: FairwashingAudit,
    additional_metrics: Dict[str, float] = None
) -> Dict[str, Any]
```

**Paramètres :**
- `explanation` : Explication à évaluer
- `uncertainty_metrics` : Métriques d'incertitude
- `fairwashing_audit` : Résultats de l'audit de fairwashing
- `additional_metrics` : Métriques supplémentaires (optionnel)

**Retourne :**
- `Dict[str, Any]` : Rapport de confiance complet

```python
def apply_to_explanation(
    self,
    explanation: Dict[str, Any],
    confidence_report: Dict[str, Any]
) -> Dict[str, Any]
```

**Paramètres :**
- `explanation` : Explication à enrichir
- `confidence_report` : Rapport de confiance à appliquer

**Retourne :**
- `Dict[str, Any]` : Explication enrichie avec le rapport de confiance

##### `TrustScore`

```python
class TrustScore:
    global_trust: float
    trust_level: TrustLevel
    uncertainty_trust: float
    fairwashing_trust: float
    consistency_trust: float
    robustness_trust: float
    trust_factors: List[str]
```

**Attributs :**
- `global_trust` : Score global de confiance (0-1)
- `trust_level` : Niveau de confiance
- `uncertainty_trust` : Score de confiance lié à l'incertitude
- `fairwashing_trust` : Score de confiance lié au fairwashing
- `consistency_trust` : Score de confiance lié à la cohérence
- `robustness_trust` : Score de confiance lié à la robustesse
- `trust_factors` : Facteurs influençant le score de confiance

##### `TrustLevel`

```python
class TrustLevel(Enum):
    VERY_LOW = "very-low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very-high"
```

**Valeurs :**
- `VERY_LOW` : Confiance très faible
- `LOW` : Confiance faible
- `MODERATE` : Confiance modérée
- `HIGH` : Confiance élevée
- `VERY_HIGH` : Confiance très élevée

## Formatters

### TrustFormatterMixin

```python
class TrustFormatterMixin:
    def _process_trust_metrics(self, explanation: Dict[str, Any]) -> Dict[str, Any]
```

**Paramètres :**
- `explanation` : Explication avec métriques de confiance

**Retourne :**
- `Dict[str, Any]` : Données de confiance formatées pour affichage

### TrustHTMLReportGenerator

```python
class TrustHTMLReportGenerator(HTMLReportGenerator, TrustFormatterMixin):
    def _generate_content(self, html_content: str, content: ReportContent) -> str
```

**Paramètres :**
- `html_content` : Contenu HTML de base
- `content` : Contenu du rapport

**Retourne :**
- `str` : Contenu HTML enrichi avec les métriques de confiance

### TrustPDFReportGenerator

```python
class TrustPDFReportGenerator(PDFReportGenerator, TrustFormatterMixin):
    def _generate_content(self, pdf: XPLIAReport, content: ReportContent) -> None
```

**Paramètres :**
- `pdf` : Objet PDF
- `content` : Contenu du rapport

## Exemples d'utilisation

### Quantification d'incertitude

```python
from xplia.explainers import UncertaintyQuantifier, UncertaintyType

# Création du quantificateur
quantifier = UncertaintyQuantifier(
    n_bootstrap_samples=100,
    confidence_level=0.95,
    methods=["bootstrap", "ensemble"]
)

# Quantification
metrics = quantifier.quantify_uncertainty(
    explanation=explanation,
    explainer=explainer,
    X=X_test,
    uncertainty_types=[UncertaintyType.ALEATORIC, UncertaintyType.EPISTEMIC]
)

# Accès aux résultats
global_uncertainty = metrics.global_uncertainty
feature_uncertainties = metrics.feature_uncertainties
```

### Détection de fairwashing

```python
from xplia.explainers import FairwashingDetector, FairwashingType

# Création du détecteur
detector = FairwashingDetector(
    sensitive_features=["gender", "race"],
    detection_threshold=0.7,
    methods=["consistency", "sensitivity"]
)

# Détection
audit = detector.detect_fairwashing(
    explanation=explanation,
    X=X_test,
    fairwashing_types=[FairwashingType.FEATURE_MASKING]
)

# Accès aux résultats
fairwashing_score = audit.fairwashing_score
detected_types = audit.detected_types
```

### Génération de rapport de confiance

```python
from xplia.explainers import ConfidenceReport

# Création du générateur
reporter = ConfidenceReport(
    uncertainty_weight=0.4,
    fairwashing_weight=0.3,
    consistency_weight=0.2,
    robustness_weight=0.1
)

# Génération du rapport
report = reporter.generate_report(
    explanation=explanation,
    uncertainty_metrics=metrics,
    fairwashing_audit=audit
)

# Application à l'explication
enriched_explanation = reporter.apply_to_explanation(explanation, report)
```

### Utilisation des formatters

```python
from xplia.compliance.formatters.html_trust_formatter import TrustHTMLReportGenerator
from xplia.compliance.formatters.pdf_trust_formatter import TrustPDFReportGenerator
from xplia.compliance.report_base import ReportConfig, ReportContent

# Configuration
config = ReportConfig(template_name="standard", language="fr")

# Création des générateurs
html_generator = TrustHTMLReportGenerator(config)
pdf_generator = TrustPDFReportGenerator(config)

# Contenu du rapport
content = ReportContent(
    title="Rapport d'Explication",
    explanations=[enriched_explanation]
)

# Génération des rapports
html_generator.generate(content, "rapport.html")
pdf_generator.generate(content, "rapport.pdf")
```

## Constantes et valeurs par défaut

### Seuils de confiance

```python
# Seuils pour les niveaux de confiance
TRUST_LEVEL_THRESHOLDS = {
    TrustLevel.VERY_LOW: 0.0,
    TrustLevel.LOW: 0.3,
    TrustLevel.MODERATE: 0.5,
    TrustLevel.HIGH: 0.7,
    TrustLevel.VERY_HIGH: 0.9
}

# Étiquettes des niveaux de confiance
TRUST_LEVEL_LABELS = {
    TrustLevel.VERY_LOW: "Très faible",
    TrustLevel.LOW: "Faible",
    TrustLevel.MODERATE: "Modérée",
    TrustLevel.HIGH: "Élevée",
    TrustLevel.VERY_HIGH: "Très élevée"
}
```

### Méthodes d'estimation d'incertitude

```python
# Méthodes disponibles pour l'estimation d'incertitude
UNCERTAINTY_METHODS = [
    "bootstrap",
    "ensemble",
    "sensitivity",
    "variance",
    "bayesian",
    "monte_carlo",
    "jackknife"
]
```

### Méthodes de détection de fairwashing

```python
# Méthodes disponibles pour la détection de fairwashing
FAIRWASHING_METHODS = [
    "consistency",
    "sensitivity",
    "counterfactual",
    "statistical",
    "adversarial",
    "perturbation",
    "benchmark"
]
```

## Codes d'erreur

```python
# Codes d'erreur pour les modules de confiance
ERROR_CODES = {
    "E001": "Données insuffisantes pour l'estimation d'incertitude",
    "E002": "Méthode d'estimation non prise en charge",
    "E003": "Feature sensible non trouvée dans les données",
    "E004": "Seuil de détection hors limites (0-1)",
    "E005": "Explication incompatible avec le format attendu",
    "E006": "Métriques d'incertitude manquantes ou invalides",
    "E007": "Résultats d'audit de fairwashing manquants ou invalides"
}
```
