# Guide de Conformité Réglementaire XPLIA

Ce guide explique comment utiliser les modules de conformité réglementaire de XPLIA pour assurer que vos systèmes d'IA respectent les réglementations comme le RGPD (GDPR) et l'AI Act européen.

## Table des matières

- [Introduction](#introduction)
- [Conformité RGPD/GDPR](#conformité-rgpdgdpr)
  - [Configuration rapide](#configuration-rapide-rgpd)
  - [Analyse approfondie](#analyse-approfondie-rgpd)
  - [Gestion des droits des individus](#gestion-des-droits-des-individus)
- [Conformité AI Act](#conformité-ai-act)
  - [Configuration rapide](#configuration-rapide-ai-act)
  - [Évaluation des risques](#évaluation-des-risques)
  - [Documentation technique](#documentation-technique)
- [Génération de rapports de conformité](#génération-de-rapports-de-conformité)
  - [Rapports unifiés](#rapports-unifiés)
  - [Formats de sortie](#formats-de-sortie)
- [Intégration avec l'explicabilité](#intégration-avec-lexplicabilité)
- [Bonnes pratiques](#bonnes-pratiques)

## Introduction

Les modules de conformité réglementaire de XPLIA fournissent des outils pour:

1. Auditer la conformité des systèmes d'IA aux principales réglementations
2. Documenter les traitements de données et les décisions algorithmiques
3. Évaluer les risques et proposer des mesures d'atténuation
4. Générer des rapports détaillés pour les autorités ou l'audit interne

## Conformité RGPD/GDPR

### Configuration rapide RGPD

```python
import xplia as xp

# Vérifier rapidement la conformité RGPD d'un modèle
gdpr_result = xp.analyze_gdpr_compliance(
    model=my_model,
    data=training_data,
    data_categories=['personal_data', 'financial_data']
)

# Afficher un résumé des résultats
print(gdpr_result.summary())

# Exporter les résultats pour documentation
gdpr_documentation = gdpr_result.export_documentation()
```

### Analyse approfondie RGPD

Pour une analyse plus détaillée, vous pouvez configurer des paramètres supplémentaires :

```python
import xplia as xp
from xplia.compliance.explanation_rights import GDPRComplianceManager, DataCategory, LegalBasis

# Créer un gestionnaire de conformité RGPD personnalisé
gdpr_manager = GDPRComplianceManager()

# Définir le DPO (Data Protection Officer)
gdpr_manager.set_dpo_contact({
    'name': 'Marie Dupont',
    'email': 'dpo@entreprise.com',
    'phone': '+33 1 23 45 67 89'
})

# Enregistrer un traitement de données
gdpr_manager.data_processing_registry.register_processing(
    name="Système de scoring crédit", 
    description="Évaluation du risque de crédit des demandeurs",
    categories=[DataCategory.FINANCIAL, DataCategory.PERSONAL],
    legal_basis=LegalBasis.LEGITIMATE_INTEREST,
    retention_period=36,  # mois
    data_subject_rights=['access', 'explanation', 'rectification']
)

# Configurer une DPIA (Data Protection Impact Assessment)
dpia = gdpr_manager.create_dpia(
    system_name="Système de scoring crédit",
    description="Système automatisé d'évaluation du risque crédit",
    processing_purpose="Évaluer la solvabilité des demandeurs",
    necessity_assessment="Essentiel pour minimiser les risques financiers"
)

dpia.add_risk({
    "description": "Discrimination indirecte basée sur des proxies",
    "likelihood": "medium",
    "impact": "high",
    "mitigation": "Tests réguliers pour détecter les biais et audits indépendants"
})

# Effectuer l'analyse complète
result = gdpr_manager.analyze(model, data, include_dpia=True)

# Exporter la documentation pour les autorités
documentation = gdpr_manager.export_documentation(format='pdf')
```

### Gestion des droits des individus

```python
# Enregistrer une demande d'explication d'un utilisateur
gdpr_manager.request_log.add(
    user_id="user123",
    data_requested={"input_features": user_data},
    explanation_provided=explanation_result
)

# Gérer une demande d'accès aux données
access_request = gdpr_manager.handle_subject_request(
    request_type="access",
    user_id="user123"
)

# Gérer une demande de rectification
rectification_request = gdpr_manager.handle_subject_request(
    request_type="rectification",
    user_id="user123",
    correction_data={"income": 52000}
)

# Vérifier l'historique des demandes d'un utilisateur
history = gdpr_manager.get_user_request_history("user123")
```

## Conformité AI Act

### Configuration rapide AI Act

```python
import xplia as xp

# Vérifier rapidement la conformité AI Act d'un modèle
ai_act_result = xp.analyze_ai_act_compliance(
    model=my_model,
    use_case='credit_scoring',
    risk_level='high'  # Systèmes à haut risque selon l'AI Act
)

# Afficher les résultats
print(ai_act_result.summary())
print(ai_act_result.risk_assessment())
```

### Évaluation des risques

```python
import xplia as xp
from xplia.compliance.ai_act import AIActComplianceManager, RiskLevel, AISystemCategory

# Créer un gestionnaire AI Act personnalisé
ai_act_manager = AIActComplianceManager()

# Définir la catégorie du système et son cas d'utilisation
ai_act_manager.set_system_category(AISystemCategory.HIGH_RISK)
ai_act_manager.set_use_case('healthcare_diagnosis')

# Ajouter un risque identifié
ai_act_manager.add_risk({
    'risk_id': 'risk001',
    'description': 'Erreur de diagnostic avec impact médical',
    'risk_level': 'high',
    'impact_areas': ['safety', 'health'],
    'likelihood': 0.3,
    'potential_harm': 'Traitement inadapté pour le patient'
})

# Ajouter une mesure d'atténuation
ai_act_manager.add_risk_mitigation({
    'mitigation_id': 'mit001',
    'target_risk_id': 'risk001',
    'description': 'Validation systématique par un médecin',
    'implementation_status': 'implemented',
    'effectiveness': 0.8
})

# Effectuer l'audit complet
compliance_result = ai_act_manager.audit(model, data)

# Exporter les résultats
technical_documentation = ai_act_manager.generate_technical_documentation()
```

### Documentation technique

```python
# Générer la documentation technique requise par l'AI Act
tech_docs = ai_act_manager.generate_technical_documentation(
    include_sections=[
        'system_description',
        'architecture',
        'development_details',
        'quality_control',
        'data_governance',
        'post_deployment'
    ]
)

# Ajouter des informations personnalisées
ai_act_manager.update_technical_documentation('system_description', """
Le système d'IA XYZ est conçu pour aider au diagnostic radiologique
des anomalies pulmonaires sur des images de tomodensitométrie.
Le système n'est pas conçu pour remplacer l'expertise médicale mais
pour servir d'outil d'aide à la décision.
""")

# Vérifier les exigences applicables
requirements = ai_act_manager.get_applicable_requirements()
compliant_reqs = [r for r in requirements if r.get('status') == 'compliant']
print(f"Conformité: {len(compliant_reqs)}/{len(requirements)} exigences respectées")
```

## Génération de rapports de conformité

### Rapports unifiés

XPLIA permet de générer des rapports unifiés incluant les informations de conformité RGPD et AI Act :

```python
import xplia as xp

# Générer un rapport unifié dans plusieurs formats
reports = xp.generate_compliance_report(
    output_formats=['pdf', 'html'], 
    output_path='./compliance_reports/report_2023',
    include_visualizations=True
)

print(f"Rapport HTML: {reports['html']}")
print(f"Rapport PDF: {reports['pdf']}")

# Avec des données personnalisées
custom_reports = xp.generate_compliance_report(
    gdpr_data=my_gdpr_manager.export_data(),
    ai_act_data=my_ai_act_manager.export_data(),
    output_formats=['json'],
    include_risk_assessment=True,
    include_dpia=True
)
```

### Formats de sortie

Les rapports de conformité peuvent être générés dans différents formats :

- **HTML**: Format interactif avec visualisations et navigation facile
- **PDF**: Format documentaire officiel pour les autorités
- **JSON**: Format d'échange pour l'intégration avec d'autres systèmes
- **Markdown**: Format texte léger pour la documentation technique

## Intégration avec l'explicabilité

Les modules de conformité s'intègrent parfaitement avec les fonctionnalités d'explicabilité de XPLIA :

```python
import xplia as xp

# Analyser et expliquer un modèle
model, _ = xp.load_model("credit_scoring_model.pkl")
explanation = xp.explain_model(model, X_test)

# Vérifier la conformité de l'explication
compliance = xp.check_compliance(
    model=model, 
    data=X_test,
    explanation_result=explanation,
    regulations=['gdpr', 'ai_act']
)

# Générer un rapport combinant explicabilité et conformité
report = xp.generate_report(
    explanation_result=explanation,
    compliance_result=compliance,
    format='html',
    output_path='report.html'
)
```

## Bonnes pratiques

Pour une conformité optimale, suivez ces recommandations :

1. **Approche dès la conception**: Intégrez la conformité dès les premières phases du développement
2. **Documentation continue**: Documentez toutes les décisions importantes affectant les droits des personnes
3. **Audit régulier**: Auditez régulièrement vos systèmes pour maintenir la conformité
4. **Mise à jour**: Mettez à jour vos analyses de risque lors des modifications du système
5. **Transparence**: Utilisez les explications générées pour communiquer clairement avec les utilisateurs

---

Pour plus d'informations, consultez la documentation complète de l'API ou contactez notre équipe de support.
