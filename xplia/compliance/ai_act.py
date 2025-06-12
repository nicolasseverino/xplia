"""
Support de l'AI Act européen pour XPLIA
=======================================

Ce module fournit des outils complets pour la conformité avec l'AI Act européen :
- Gestion des catégories de risque (inacceptable, élevé, limité, minimal)
- Documentation automatique des décisions algorithmiques
- Suivi des logs et auditabilité des systèmes d'IA
- Gestion de la transparence et de l'information des utilisateurs
- Évaluation et atténuation des risques
- Génération de documentation technique conforme

Références normatives :
- Règlement (UE) 2024/XXX du Parlement européen et du Conseil (AI Act)
- EN 303 645 - Cybersécurité pour l'IoT
- ISO/IEC 42001 - Systèmes de management de l'IA
"""

from typing import Any, Dict, List, Optional, Union
import datetime
import json
import hashlib
import os
import uuid
from enum import Enum, auto
from ..core.base import ConfigurableMixin

class AIRiskCategory(Enum):
    """Catégories de risque définies par l'AI Act européen"""
    UNACCEPTABLE = "unacceptable"
    HIGH = "high"
    MEDIUM = "medium"
    LIMITED = "limited"
    MINIMAL = "minimal"
    
    @classmethod
    def get_requirements(cls, category):
        """Retourne les exigences associées à une catégorie de risque."""
        requirements = {
            cls.UNACCEPTABLE: ["Interdit par l'AI Act"],
            cls.HIGH: [
                "Analyse d'impact obligatoire", 
                "Supervision humaine", 
                "Robustesse technique", 
                "Précision et mitigation des biais",
                "Enregistrement des activités",
                "Documentation technique détaillée",
                "Information des utilisateurs"
            ],
            cls.MEDIUM: [
                "Analyse d'impact recommandée", 
                "Supervision humaine", 
                "Documentation technique",
                "Information des utilisateurs"
            ],
            cls.LIMITED: [
                "Documentation technique de base",
                "Information des utilisateurs"
            ],
            cls.MINIMAL: [
                "Conformité aux bonnes pratiques"
            ]
        }
        return requirements.get(category, ["Catégorie non reconnue"])

class ModelUsageIntent(Enum):
    """Intention d'utilisation du modèle d'IA selon l'AI Act"""
    CRITICAL_INFRASTRUCTURE = auto()  # Infrastructure critique (sécurité, santé, etc.)
    EDUCATION = auto()                # Éducation ou formation professionnelle
    EMPLOYMENT = auto()               # Emploi, gestion des travailleurs
    ESSENTIAL_SERVICES = auto()       # Accès à des services essentiels
    LAW_ENFORCEMENT = auto()          # Application de la loi
    MIGRATION = auto()                # Migration, asile et contrôle aux frontières
    ADMINISTRATION_JUSTICE = auto()   # Administration de la justice
    DEMOCRATIC_PROCESS = auto()       # Processus démocratiques
    GENERAL_PURPOSE = auto()          # IA à usage général
    RESEARCH = auto()                 # Recherche scientifique
    OTHER = auto()                    # Autre usage
    
    @classmethod
    def get_risk_category(cls, intent):
        """Détermine la catégorie de risque par défaut selon l'intention d'utilisation."""
        high_risk = [
            cls.CRITICAL_INFRASTRUCTURE, cls.EDUCATION, cls.EMPLOYMENT,
            cls.ESSENTIAL_SERVICES, cls.LAW_ENFORCEMENT, cls.MIGRATION,
            cls.ADMINISTRATION_JUSTICE, cls.DEMOCRATIC_PROCESS
        ]
        
        if intent in high_risk:
            return AIRiskCategory.HIGH
        elif intent == cls.GENERAL_PURPOSE:
            return AIRiskCategory.MEDIUM
        elif intent == cls.RESEARCH:
            return AIRiskCategory.LIMITED
        else:
            return AIRiskCategory.MEDIUM  # Par défaut

class AuditRecord:
    """Enregistrement d'audit conforme à l'AI Act"""
    def __init__(self, action_type, details, user_id=None, model_id=None):
        self.timestamp = datetime.datetime.now().isoformat()
        self.action_type = action_type
        self.details = details
        self.user_id = user_id
        self.model_id = model_id
        self.record_id = str(uuid.uuid4())
        
    def to_dict(self):
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "action_type": self.action_type,
            "details": self.details,
            "user_id": self.user_id,
            "model_id": self.model_id
        }

class AIActComplianceManager(ConfigurableMixin):
    """
    Gère la conformité avec l'AI Act européen pour XPLIA.
    
    Cette classe fournit les fonctionnalités suivantes :
    - Catégorisation des risques selon l'AI Act
    - Journal d'audit des décisions algorithmiques
    - Génération de documentation technique
    - Évaluation de conformité
    - Mesures de transparence pour les utilisateurs finaux
    """
    def __init__(self, risk_category=None, model_usage_intent=None, model_metadata=None):
        super().__init__()
        
        # Déterminer la catégorie de risque
        if risk_category:
            self.risk_category = risk_category
        elif model_usage_intent:
            self.risk_category = ModelUsageIntent.get_risk_category(model_usage_intent)
        else:
            self.risk_category = AIRiskCategory.MEDIUM
            
        # Initialisation
        self.model_metadata = model_metadata or {}
        self.model_usage_intent = model_usage_intent
        self.decision_log = []
        self.audit_trail = []
        self.technical_documentation = {}
        self.risk_assessment = {}
        
        # Configuration par défaut
        self._config.set_default('storage_path', os.path.join(os.getcwd(), 'ai_act_logs'))
        self._config.set_default('max_log_entries', 1000)
        self._config.set_default('export_format', 'json')
        
        # Générer un identifiant unique pour ce gestionnaire
        self.manager_id = str(uuid.uuid4())
        
        # Enregistrer l'initialisation dans l'audit
        self._add_audit_record('initialization', {
            'risk_category': self.risk_category.value if isinstance(self.risk_category, Enum) else self.risk_category,
            'model_usage_intent': self.model_usage_intent.name if self.model_usage_intent else None,
            'manager_id': self.manager_id
        })
    
    def _add_audit_record(self, action_type, details, user_id=None, model_id=None):
        """Ajoute un enregistrement à l'audit trail"""
        record = AuditRecord(action_type, details, user_id, model_id)
        self.audit_trail.append(record.to_dict())
        return record.record_id
    
    def log_decision(self, input_data: Any, output: Any, explanation: Any, user_id: str = None, model_id: str = None):
        """Journalise une décision algorithmique pour conformité à l'AI Act"""
        # Générer une empreinte des données d'entrée pour référence tout en préservant la confidentialité
        try:
            input_hash = hashlib.sha256(str(input_data).encode()).hexdigest()
        except:
            input_hash = "non_hashable_input"
            
        decision_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        # Créer un enregistrement détaillé de la décision
        decision_record = {
            "decision_id": decision_id,
            "timestamp": timestamp,
            "user_id": user_id,
            "model_id": model_id,
            "input_hash": input_hash,
            "input_data": input_data,  # Note: dans un système de production, considérer l'anonymisation
            "output": output,
            "explanation": explanation,
            "risk_category": self.risk_category.value if isinstance(self.risk_category, Enum) else self.risk_category,
            "requirements": AIRiskCategory.get_requirements(self.risk_category) if isinstance(self.risk_category, Enum) else []
        }
        
        self.decision_log.append(decision_record)
        
        # Limiter la taille du journal si nécessaire
        max_entries = self._config.get('max_log_entries')
        if len(self.decision_log) > max_entries:
            self.decision_log = self.decision_log[-max_entries:]
            
        # Ajouter également un enregistrement d'audit
        self._add_audit_record('decision', {
            'decision_id': decision_id,
            'user_id': user_id,
            'model_id': model_id,
            'timestamp': timestamp,
            'risk_category': self.risk_category.value if isinstance(self.risk_category, Enum) else self.risk_category
        }, user_id, model_id)
        
        return decision_id
    
    def export_decision_log(self, format=None, path=None) -> Union[List, str]:
        """Exporte le journal des décisions dans le format spécifié"""
        format = format or self._config.get('export_format')
        path = path or self._config.get('storage_path')
        
        if format.lower() == 'json':
            if path:
                os.makedirs(path, exist_ok=True)
                file_path = os.path.join(path, f"decisions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.decision_log, f, ensure_ascii=False, indent=2, default=str)
                return file_path
            else:
                return json.dumps(self.decision_log, ensure_ascii=False, indent=2, default=str)
        else:
            # Format par défaut - retourner la liste des décisions
            return self.decision_log
    
    def export_audit_trail(self, format=None, path=None) -> Union[List, str]:
        """Exporte la piste d'audit dans le format spécifié"""
        format = format or self._config.get('export_format')
        path = path or self._config.get('storage_path')
        
        if format.lower() == 'json':
            if path:
                os.makedirs(path, exist_ok=True)
                file_path = os.path.join(path, f"audit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.audit_trail, f, ensure_ascii=False, indent=2, default=str)
                return file_path
            else:
                return json.dumps(self.audit_trail, ensure_ascii=False, indent=2, default=str)
        else:
            # Format par défaut - retourner la liste des enregistrements d'audit
            return self.audit_trail
    
    def get_risk_category(self) -> Union[AIRiskCategory, str]:
        """Retourne la catégorie de risque actuelle"""
        return self.risk_category
    
    def set_risk_category(self, category: Union[AIRiskCategory, str]):
        """Définit la catégorie de risque"""
        old_category = self.risk_category
        self.risk_category = category
        
        # Journaliser le changement de catégorie de risque
        self._add_audit_record('risk_category_change', {
            'old_category': old_category.value if isinstance(old_category, Enum) else old_category,
            'new_category': category.value if isinstance(category, Enum) else category
        })
        
        return True
        
    def perform_risk_assessment(self, model_data=None, training_data=None, validation_data=None):
        """
        Réalise une évaluation des risques conforme à l'AI Act.
        
        Args:
            model_data: Métadonnées du modèle ou modèle lui-même
            training_data: Données d'entraînement (facultatif)
            validation_data: Données de validation (facultatif)
            
        Returns:
            Dict: Résultats de l'évaluation des risques
        """
        # Initialisation de l'évaluation des risques
        risk_assessment = {
            "timestamp": datetime.datetime.now().isoformat(),
            "risk_category": self.risk_category.value if isinstance(self.risk_category, Enum) else self.risk_category,
            "model_id": self.model_metadata.get("model_id", "unknown"),
            "assessment_id": str(uuid.uuid4()),
            "risks": [],
            "mitigations": [],
            "compliance_score": 0.0,
            "requirements": AIRiskCategory.get_requirements(self.risk_category) if isinstance(self.risk_category, Enum) else []
        }
        
        # Définir les risques en fonction de la catégorie
        if isinstance(self.risk_category, Enum):
            if self.risk_category == AIRiskCategory.HIGH or self.risk_category == AIRiskCategory.MEDIUM:
                # Risques spécifiques aux systèmes à haut risque
                risk_assessment["risks"].extend([
                    {"id": "bias_fairness", "name": "Biais et équité", "description": "Risque de discrimination algorithmique", "level": "high"},
                    {"id": "transparency", "name": "Transparence", "description": "Manque d'explicabilité des décisions", "level": "medium"},
                    {"id": "data_quality", "name": "Qualité des données", "description": "Données d'entraînement biaises ou incomplètes", "level": "high"},
                    {"id": "human_oversight", "name": "Supervision humaine", "description": "Supervision humaine insuffisante", "level": "medium"},
                    {"id": "documentation", "name": "Documentation", "description": "Documentation technique incomplète", "level": "medium"},
                ])
                
                # Proposer des mesures d'atténuation
                risk_assessment["mitigations"].extend([
                    {"risk_id": "bias_fairness", "action": "Audit régulier des biais", "status": "to_implement"},
                    {"risk_id": "transparency", "action": "Implémenter des expliqueurs locaux et globaux", "status": "to_implement"},
                    {"risk_id": "data_quality", "action": "Mettre en place une gouvernance des données", "status": "to_implement"},
                    {"risk_id": "human_oversight", "action": "Définir un processus de supervision humaine", "status": "to_implement"},
                    {"risk_id": "documentation", "action": "Générer une documentation technique complète", "status": "to_implement"},
                ])
            elif self.risk_category == AIRiskCategory.LIMITED:
                # Risques limités
                risk_assessment["risks"].extend([
                    {"id": "transparency", "name": "Transparence", "description": "Information minimale de l'utilisateur", "level": "low"},
                    {"id": "documentation", "name": "Documentation", "description": "Documentation de base", "level": "low"},
                ])
                
                # Proposer des mesures d'atténuation
                risk_assessment["mitigations"].extend([
                    {"risk_id": "transparency", "action": "Informer l'utilisateur de l'utilisation d'IA", "status": "to_implement"},
                    {"risk_id": "documentation", "action": "Créer une documentation de base", "status": "to_implement"},
                ])
        
        # Calcul d'un score de conformité (simple pour l'instant)
        # Le score est basé sur 0 mesure mise en œuvre sur N mesures proposées
        if len(risk_assessment["mitigations"]) > 0:
            implemented = sum(1 for m in risk_assessment["mitigations"] if m["status"] == "implemented")
            risk_assessment["compliance_score"] = (implemented / len(risk_assessment["mitigations"])) * 100
        
        # Journaliser l'évaluation
        self.risk_assessment = risk_assessment
        self._add_audit_record('risk_assessment', {
            'assessment_id': risk_assessment["assessment_id"],
            'risk_count': len(risk_assessment["risks"]),
            'compliance_score': risk_assessment["compliance_score"]
        })
        
        return risk_assessment
    
    def implement_mitigation(self, risk_id, action_description=None, evidence=None):
        """
        Met en œuvre une mesure d'atténuation d'un risque identifié.
        
        Args:
            risk_id: Identifiant du risque à atténuer
            action_description: Description de l'action mise en œuvre
            evidence: Preuves de mise en œuvre (chemins de fichier, liens, etc.)
            
        Returns:
            bool: True si la mesure a été mise à jour avec succès
        """
        if not self.risk_assessment or "mitigations" not in self.risk_assessment:
            return False
            
        # Rechercher la mesure associée au risque
        updated = False
        for i, mitigation in enumerate(self.risk_assessment["mitigations"]):
            if mitigation["risk_id"] == risk_id:
                # Mettre à jour le statut et ajouter des informations sur l'action
                self.risk_assessment["mitigations"][i]["status"] = "implemented"
                self.risk_assessment["mitigations"][i]["implementation_date"] = datetime.datetime.now().isoformat()
                
                if action_description:
                    self.risk_assessment["mitigations"][i]["action_description"] = action_description
                    
                if evidence:
                    self.risk_assessment["mitigations"][i]["evidence"] = evidence
                
                updated = True
        
        # Recalculer le score de conformité
        if updated and len(self.risk_assessment["mitigations"]) > 0:
            implemented = sum(1 for m in self.risk_assessment["mitigations"] if m["status"] == "implemented")
            self.risk_assessment["compliance_score"] = (implemented / len(self.risk_assessment["mitigations"])) * 100
            
            # Journaliser la mise en œuvre de la mesure
            self._add_audit_record('mitigation_implemented', {
                'risk_id': risk_id,
                'action': action_description,
                'new_compliance_score': self.risk_assessment["compliance_score"]
            })
            
        return updated
    
    def generate_technical_documentation(self, output_format="markdown", output_path=None):
        """
        Génère la documentation technique conforme aux exigences de l'AI Act.
        
        Args:
            output_format: Format de sortie (markdown, html, json)
            output_path: Chemin de sauvegarde du fichier
            
        Returns:
            Union[str, Dict]: Documentation technique au format demandé
        """
        if not self.model_metadata:
            self.model_metadata = {"model_id": "unknown", "type": "unknown", "version": "unknown"}
            
        # Création de la structure de documentation technique
        tech_doc = {
            "title": f"Documentation technique - {self.model_metadata.get('type', 'Modèle')} IA",
            "model_id": self.model_metadata.get("model_id", "unknown"),
            "version": self.model_metadata.get("version", "1.0.0"),
            "risk_category": self.risk_category.value if isinstance(self.risk_category, Enum) else self.risk_category,
            "created_date": datetime.datetime.now().isoformat(),
            "generated_by": "XPLIA AI Act Compliance Manager",
            "sections": [
                {
                    "title": "1. Description générale du système d'IA",
                    "content": self.model_metadata.get("description", "[Insérer description du système]")
                },
                {
                    "title": "2. Aperçu de la conception",
                    "content": self.model_metadata.get("design_overview", "[Décrire l'architecture générale du système]")
                },
                {
                    "title": "3. Données et entraînement",
                    "content": self.model_metadata.get("training_data_description", "[Décrire les données utilisées pour l'entraînement]"),
                    "subsections": [
                        {
                            "title": "3.1. Sources de données",
                            "content": self.model_metadata.get("data_sources", "[Lister les sources de données]")
                        },
                        {
                            "title": "3.2. Prétraitement des données",
                            "content": self.model_metadata.get("data_preprocessing", "[Décrire le prétraitement appliqué]")
                        }
                    ]
                },
                {
                    "title": "4. Évaluation des risques",
                    "content": "Résultats de l'évaluation des risques effectuée conformément à l'AI Act.",
                    "data": self.risk_assessment
                },
                {
                    "title": "5. Mesures de conformité",
                    "content": "Mesures mises en place pour assurer la conformité avec l'AI Act.",
                    "requirements": AIRiskCategory.get_requirements(self.risk_category) if isinstance(self.risk_category, Enum) else []
                },
                {
                    "title": "6. Performances et métriques",
                    "content": self.model_metadata.get("performance", "[Insérer métriques de performance]")
                },
                {
                    "title": "7. Supervision humaine",
                    "content": self.model_metadata.get("human_oversight", "[Décrire les mesures de supervision humaine]")
                },
            ]
        }
        
        # Enregistrer la documentation
        self.technical_documentation = tech_doc
        
        # Générer le format demandé
        if output_format == "json":
            result = json.dumps(tech_doc, ensure_ascii=False, indent=2)
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result)
            return result
            
        elif output_format == "markdown":
            md = f"# {tech_doc['title']}\n\n"
            md += f"**Identifiant du modèle:** {tech_doc['model_id']}  \n"
            md += f"**Version:** {tech_doc['version']}  \n"
            md += f"**Catégorie de risque:** {tech_doc['risk_category']}  \n"
            md += f"**Date de création:** {tech_doc['created_date']}  \n\n"
            
            for section in tech_doc["sections"]:
                md += f"## {section['title']}\n\n"
                md += f"{section['content']}\n\n"
                
                if "subsections" in section:
                    for subsection in section["subsections"]:
                        md += f"### {subsection['title']}\n\n"
                        md += f"{subsection['content']}\n\n"
                
                if "data" in section and section["data"] and "risks" in section["data"]:
                    md += "### Risques identifiés\n\n"
                    for risk in section["data"]["risks"]:
                        md += f"- **{risk['name']}** ({risk['level']}): {risk['description']}\n"
                    md += "\n"
                
                if "requirements" in section and section["requirements"]:
                    md += "### Exigences applicables\n\n"
                    for req in section["requirements"]:
                        md += f"- {req}\n"
                    md += "\n"
            
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(md)
            
            return md
            
        elif output_format == "html":
            # Version simplifiée HTML
            html = f"<!DOCTYPE html>\n<html>\n<head>\n<meta charset='utf-8'>\n<title>{tech_doc['title']}</title>\n"
            html += "<style>body{font-family:system-ui;max-width:800px;margin:0 auto;padding:20px}h1,h2,h3{color:#345;}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}</style>\n"
            html += "</head>\n<body>\n"
            html += f"<h1>{tech_doc['title']}</h1>\n"
            html += f"<p><strong>Identifiant du modèle:</strong> {tech_doc['model_id']}<br>"
            html += f"<strong>Version:</strong> {tech_doc['version']}<br>"
            html += f"<strong>Catégorie de risque:</strong> {tech_doc['risk_category']}<br>"
            html += f"<strong>Date de création:</strong> {tech_doc['created_date']}</p>\n"
            
            for section in tech_doc["sections"]:
                html += f"<h2>{section['title']}</h2>\n"
                html += f"<p>{section['content']}</p>\n"
                
                if "subsections" in section:
                    for subsection in section["subsections"]:
                        html += f"<h3>{subsection['title']}</h3>\n"
                        html += f"<p>{subsection['content']}</p>\n"
                
                if "data" in section and section["data"] and "risks" in section["data"]:
                    html += "<h3>Risques identifiés</h3>\n<ul>\n"
                    for risk in section["data"]["risks"]:
                        html += f"<li><strong>{risk['name']}</strong> ({risk['level']}): {risk['description']}</li>\n"
                    html += "</ul>\n"
                
                if "requirements" in section and section["requirements"]:
                    html += "<h3>Exigences applicables</h3>\n<ul>\n"
                    for req in section["requirements"]:
                        html += f"<li>{req}</li>\n"
                    html += "</ul>\n"
            
            html += "</body>\n</html>"
            
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html)
            
            return html
        
        # Format par défaut
        return tech_doc
