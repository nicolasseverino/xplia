"""
API principale de XPLIA
======================

Ce module fournit une interface simple et intuitive pour utiliser
toutes les fonctionnalités de XPLIA sans avoir à comprendre les détails
d'implémentation sous-jacents.
"""

from ..core.factory import ModelFactory, ExplainerFactory
from ..core.config import ConfigManager
from ..core.base import ExplainabilityMethod, AudienceLevel
from ..compliance.explanation_rights import GDPRComplianceManager
from ..compliance.ai_act import AIActComplianceManager
from ..compliance.compliance_report import ComplianceReportGenerator

# API Export
__all__ = [
    'load_model', 
    'explain_model', 
    'explain_prediction',
    'create_visualizer', 
    'generate_report',
    'set_config',
    'check_compliance',
    'generate_compliance_report',
    'export_audit_trail',
    'export_decision_log',
    'analyze_gdpr_compliance',
    'analyze_ai_act_compliance'
]

def load_model(model_path=None, model_type=None, model_instance=None, **kwargs):
    """
    Charge un modèle à partir d'un fichier ou utilise un modèle préexistant.
    
    Args:
        model_path (str, optional): Chemin vers un fichier de modèle sauvegardé.
        model_type (str, optional): Type de modèle ('sklearn', 'tensorflow', 'pytorch', etc.).
        model_instance (object, optional): Instance de modèle déjà chargée.
        **kwargs: Arguments additionnels pour le chargement du modèle.
        
    Returns:
        tuple: (model, model_metadata) - Le modèle chargé et ses métadonnées.
        
    Examples:
        >>> import lumia as lm
        >>> model, metadata = lm.load_model("model.pkl")
        >>> model, metadata = lm.load_model(model_instance=my_model)
    """
    factory = ModelFactory()
    return factory.load_model(
        model_path=model_path, 
        model_type=model_type, 
        model_instance=model_instance, 
        **kwargs
    )

# Managers de conformité globaux
_gdpr_manager = GDPRComplianceManager()
_ai_act_manager = AIActComplianceManager()


def explain_model(model, data, target=None, features=None, 
                 method=ExplainabilityMethod.UNIFIED, audience_level=AudienceLevel.TECHNICAL,
                 user_id=None, **kwargs):
    """
    Génère des explications pour un modèle entier.
    
    Args:
        model: Modèle à expliquer (ou chemin vers un modèle).
        data: Données utilisées pour expliquer le modèle.
        target: Variable cible (optionnelle pour certains types d'explications).
        features (list, optional): Noms des caractéristiques.
        method (ExplainabilityMethod): Méthode d'explicabilité à utiliser.
        audience_level (AudienceLevel): Niveau d'audience ciblé.
        **kwargs: Paramètres supplémentaires pour l'explainer.
    
    Returns:
        ExplanationResult: Résultat de l'explication
        
    Examples:
        >>> import lumia as lm
        >>> result = lm.explain_model(model, X_test, method=lm.ExplainabilityMethod.SHAP)
        >>> result.visualize()
    """
    # Charger le modèle s'il s'agit d'un chemin
    if isinstance(model, str):
        model, _ = load_model(model_path=model)
        
    factory = ExplainerFactory()
    explainer = factory.create_explainer(model=model, method=method, audience_level=audience_level, **kwargs)
    result = explainer.explain_model(data=data, target=target, features=features, **kwargs)
    # Log conformité RGPD et AI Act
    if user_id is not None:
        _gdpr_manager.request_log.add(user_id, data, result)
    _ai_act_manager.log_decision(data, getattr(result, 'output', None), result, user_id=user_id)
    return result

def explain_prediction(model, instance, features=None, 
                      method=ExplainabilityMethod.UNIFIED, 
                      audience_level=AudienceLevel.TECHNICAL,
                      user_id=None,
                      **kwargs):
    """
    Génère des explications pour une prédiction spécifique.
    
    Args:
        model: Modèle à expliquer (ou chemin vers un modèle).
        instance: Instance individuelle à expliquer.
        features (list, optional): Noms des caractéristiques.
        method (ExplainabilityMethod): Méthode d'explicabilité à utiliser.
        audience_level (AudienceLevel): Niveau d'audience ciblé.
        **kwargs: Paramètres supplémentaires pour l'explainer.
    
    Returns:
        ExplanationResult: Résultat de l'explication
        
    Examples:
        >>> import lumia as lm
        >>> result = lm.explain_prediction(model, X_test[0], features=feature_names)
        >>> result.visualize(type='force_plot')
    """
    # Charger le modèle s'il s'agit d'un chemin
    if isinstance(model, str):
        model, _ = load_model(model_path=model)
        
    factory = ExplainerFactory()
    explainer = factory.create_explainer(model=model, method=method, audience_level=audience_level, **kwargs)
    result = explainer.explain_prediction(instance=instance, features=features, **kwargs)
    # Log conformité RGPD et AI Act
    if user_id is not None:
        _gdpr_manager.request_log.add(user_id, instance, result)
    _ai_act_manager.log_decision(instance, getattr(result, 'output', None), result, user_id=user_id)
    return result

def create_visualizer(explanation_result, type='auto', **kwargs):
    """
    Crée un visualiseur pour les résultats d'explication.
    
    Args:
        explanation_result (ExplanationResult): Résultat d'explication à visualiser.
        type (str): Type de visualisation ('dashboard', 'force_plot', 'feature_importance', etc.).
        **kwargs: Paramètres supplémentaires pour le visualiseur.
        
    Returns:
        Visualizer: Un objet visualiseur
        
    Examples:
        >>> import lumia as lm
        >>> result = lm.explain_model(model, X_test)
        >>> viz = lm.create_visualizer(result, type='dashboard')
        >>> viz.render()
    """
    from ..visualizers import VisualizerFactory
    factory = VisualizerFactory()
    return factory.create_visualizer(
        explanation_result=explanation_result,
        viz_type=type,
        **kwargs
    )

def generate_report(explanation_result=None, format='html', output_path=None, audit_trail=None, decision_log=None, **kwargs):
    """
    Génère un rapport d'explicabilité dans le format spécifié.
    
    Args:
        explanation_result (ExplanationResult): Résultat d'explication.
        format (str): Format du rapport ('html', 'pdf', 'markdown', 'json').
        output_path (str, optional): Chemin de sortie pour le rapport.
        **kwargs: Paramètres supplémentaires pour la génération du rapport.
        
    Returns:
        str: Chemin vers le rapport généré ou contenu du rapport
        
    Examples:
        >>> import lumia as lm
        >>> result = lm.explain_model(model, X_test)
        >>> report_path = lm.generate_report(result, format='html', output_path='report.html')
    """
    # Si audit_trail/decision_log non fournis, utiliser ceux des managers globaux
    audit_trail = audit_trail if audit_trail is not None else _gdpr_manager.export_audit_trail()
    decision_log = decision_log if decision_log is not None else _ai_act_manager.export_decision_log()
    generator = ComplianceReportGenerator()
    if format == 'pdf':
        if output_path is None:
            raise ValueError("output_path requis pour le format PDF")
        generator.generate_pdf(audit_trail, decision_log, output_path)
        return output_path
    elif format == 'html':
        html = generator.generate_html(audit_trail, decision_log)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
        return html
    elif format == 'markdown':
        md = _generate_markdown_report(audit_trail, decision_log)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md)
        return md
    elif format == 'json':
        import json
        data = {'audit_trail': audit_trail, 'decision_log': decision_log}
        js = json.dumps(data, ensure_ascii=False, indent=2)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(js)
        return js
    else:
        raise ValueError(f"Format de rapport non supporté : {format}")

def _generate_markdown_report(audit_trail, decision_log):
    md = "# Rapport de conformité XPLIA\n"
    md += f"**Date** : {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    md += "## Journal des demandes d'explication (RGPD)\n"
    for entry in audit_trail:
        md += f"- {entry}\n"
    md += "\n## Journal des décisions (AI Act)\n"
    for entry in decision_log:
        md += f"- {entry}\n"
    return md


def export_audit_trail():
    """
    Exporte le journal RGPD global (audit trail).
    """
    return _gdpr_manager.export_audit_trail()

def export_decision_log():
    """
    Exporte le journal des décisions AI Act global.
    """
    return _ai_act_manager.export_decision_log()

def set_config(config=None, **kwargs):
    """
    Configure les paramètres globaux de LUMIA.
    
    Args:
        config (dict, optional): Dictionnaire de configuration.
        **kwargs: Paramètres de configuration individuels.
        
    Examples:
        >>> import lumia as lm
        >>> lm.set_config(visualization_theme='dark', cache_results=True)
    """
    manager = ConfigManager()
    
    if config is not None:
        for section, items in config.items():
            for key, value in items.items():
                manager.set(section, key, value)
    
    for key, value in kwargs.items():
        # Inférer la section basée sur le nom de la clé
        if key.startswith(('viz_', 'visualization_')):
            section = 'visualization'
            key = key.replace('viz_', '').replace('visualization_', '')
        elif key.startswith(('perf_', 'performance_')):
            section = 'performance'
            key = key.replace('perf_', '').replace('performance_', '')
        elif key.startswith(('log_', 'logging_')):
            section = 'logging'
            key = key.replace('log_', '').replace('logging_', '')
        else:
            section = 'general'
            
        manager.set(section, key, value)
    
    return manager.get_all()

def check_compliance(model=None, data=None, explanation_result=None, regulations=None, **kwargs):
    """
    Vérifie la conformité d'un modèle ou d'une explication selon diverses réglementations.
    
    Args:
        model: Modèle à vérifier.
        data: Données associées au modèle.
        explanation_result (ExplanationResult, optional): Résultat d'explication à vérifier.
        regulations (list): Liste des réglementations à vérifier ('gdpr', 'ai_act', 'hipaa', etc.).
        **kwargs: Paramètres supplémentaires pour la vérification de conformité.
        
    Returns:
        ComplianceResult: Résultat de la vérification de conformité
        
    Examples:
        >>> import lumia as lm
        >>> model, _ = lm.load_model("credit_scoring_model.pkl")
        >>> compliance = lm.check_compliance(model, X_train, regulations=['gdpr', 'ai_act'])
        >>> compliance.report()
    """
    from ..compliance import ComplianceChecker
    checker = ComplianceChecker()
    return checker.check(
        model=model,
        data=data,
        explanation_result=explanation_result,
        regulations=regulations or ['gdpr', 'ai_act'],
        **kwargs
    )

def generate_compliance_report(output_formats=None, gdpr_data=None, ai_act_data=None, output_path=None, **kwargs):
    """
    Génère un rapport de conformité réglementaire complet.
    
    Cette fonction analyse les données de conformité GDPR et AI Act disponibles
    et génère un rapport détaillé au format souhaité.
    
    Args:
        output_formats (list): Liste des formats de sortie souhaités ('pdf', 'html', 'json', etc.)
        gdpr_data (dict, optional): Données spécifiques GDPR à inclure dans le rapport
        ai_act_data (dict, optional): Données spécifiques AI Act à inclure dans le rapport
        output_path (str, optional): Chemin où sauvegarder le(s) rapport(s)
        **kwargs: Options supplémentaires pour la génération du rapport
        
    Returns:
        dict: Un dictionnaire contenant les chemins des rapports générés par format
        
    Examples:
        >>> import lumia as lm
        >>> reports = lm.generate_compliance_report(
        ...     output_formats=['pdf', 'html'],
        ...     output_path='./reports/compliance_report'
        ... )
        >>> print(f"Rapport PDF généré: {reports['pdf']}")
    """
    output_formats = output_formats or ['html']
    
    # Utiliser les données globales si non spécifiées
    if gdpr_data is None:
        gdpr_data = _gdpr_manager.export_data()
    
    if ai_act_data is None:
        ai_act_data = _ai_act_manager.export_data()
    
    # Créer et initialiser le générateur de rapport
    generator = ComplianceReportGenerator()
    generator.init_gdpr_data(gdpr_data)
    generator.init_ai_act_data(ai_act_data)
    
    # Générer les rapports dans tous les formats demandés
    results = {}
    for fmt in output_formats:
        report_path = generator.generate(fmt, output_path=output_path, **kwargs)
        results[fmt] = report_path
    
    return results

def analyze_gdpr_compliance(model=None, data=None, data_processing_records=None, dpo_contact=None, **kwargs):
    """
    Analyse la conformité GDPR/RGPD d'un modèle ou d'un système.
    
    Cette fonction vérifie et évalue la conformité d'un système IA
    aux exigences du Règlement Général sur la Protection des Données.
    
    Args:
        model: Modèle à analyser
        data: Données associées au modèle
        data_processing_records (dict, optional): Registres de traitement des données
        dpo_contact (dict, optional): Informations de contact du DPO
        **kwargs: Paramètres supplémentaires pour l'analyse
        
    Returns:
        GDPRComplianceResult: Résultat de l'analyse de conformité GDPR
        
    Examples:
        >>> import lumia as lm
        >>> gdpr_analysis = lm.analyze_gdpr_compliance(
        ...     model=my_model,
        ...     data=training_data,
        ...     data_categories=['personal_data', 'financial_data']
        ... )
        >>> print(gdpr_analysis.summary())
    """
    # Créer un manager GDPR spécifique pour cette analyse
    manager = GDPRComplianceManager()
    
    # Initialiser les registres et paramètres
    if data_processing_records:
        manager.data_processing_registry.update(data_processing_records)
    
    if dpo_contact:
        manager.set_dpo_contact(dpo_contact)
    
    # Analyser le modèle et les données
    result = manager.analyze(model, data, **kwargs)
    
    return result

def analyze_ai_act_compliance(model=None, data=None, risk_level=None, use_case=None, **kwargs):
    """
    Analyse la conformité d'un système IA selon l'AI Act européen.
    
    Cette fonction évalue la conformité d'un système d'IA aux exigences
    définies par l'AI Act européen, identifie les risques et suggère
    des mesures d'atténuation.
    
    Args:
        model: Modèle à analyser
        data: Données associées au modèle
        risk_level (str, optional): Niveau de risque prédéfini ('high', 'medium', 'low')
        use_case (str, optional): Cas d'utilisation du système ('healthcare', 'finance', etc.)
        **kwargs: Paramètres supplémentaires pour l'analyse
        
    Returns:
        AIActComplianceResult: Résultat de l'analyse de conformité AI Act
        
    Examples:
        >>> import lumia as lm
        >>> ai_act_analysis = lm.analyze_ai_act_compliance(
        ...     model=my_model,
        ...     use_case='credit_scoring',
        ...     documentation_level='detailed'
        ... )
        >>> print(ai_act_analysis.risk_assessment())
    """
    # Créer un manager AI Act spécifique pour cette analyse
    manager = AIActComplianceManager()
    
    # Définir le niveau de risque et le cas d'utilisation si spécifiés
    if risk_level:
        manager.set_risk_level(risk_level)
    
    if use_case:
        manager.set_use_case(use_case)
    
    # Analyser le modèle et les données
    result = manager.analyze(model, data, **kwargs)
    
    return result
