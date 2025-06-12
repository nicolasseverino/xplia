"""
Support du RGPD (GDPR) pour XPLIA
=================================

Ce module fournit des outils complets pour la conformité avec le RGPD européen :
- Gestion des principes fondamentaux du RGPD (licéité, loyauté, transparence, etc.)
- Documentation des traitements de données personnelles
- Gestion des droits des personnes concernées (accès, rectification, etc.)
- Aide à la réalisation d'analyses d'impact (AIPD/DPIA)
- Gestion des violations de données
- Génération de documentation conforme

Références normatives :
- Règlement (UE) 2016/679 du Parlement européen et du Conseil (RGPD)
- Lignes directrices du Comité européen de la protection des données (CEPD)
- ISO/IEC 27701 - Extension d'ISO/IEC 27001 pour la gestion des informations de privacy
"""

from typing import Any, Dict, List, Optional, Union, Set
import datetime
import json
import hashlib
import os
import uuid
from enum import Enum, auto
from ..core.base import ConfigurableMixin


class GDPRPrinciple(Enum):
    """Principes fondamentaux du RGPD (Article 5)"""
    LAWFULNESS_FAIRNESS_TRANSPARENCY = "lawfulness_fairness_transparency"
    PURPOSE_LIMITATION = "purpose_limitation"
    DATA_MINIMISATION = "data_minimisation"
    ACCURACY = "accuracy"
    STORAGE_LIMITATION = "storage_limitation"
    INTEGRITY_CONFIDENTIALITY = "integrity_confidentiality"
    ACCOUNTABILITY = "accountability"
    
    @classmethod
    def get_description(cls, principle):
        """Retourne la description d'un principe du RGPD."""
        descriptions = {
            cls.LAWFULNESS_FAIRNESS_TRANSPARENCY: 
                "Les données doivent être traitées de manière licite, loyale et transparente.",
            cls.PURPOSE_LIMITATION: 
                "Les données doivent être collectées pour des finalités déterminées, explicites et légitimes.",
            cls.DATA_MINIMISATION: 
                "Les données doivent être adéquates, pertinentes et limitées à ce qui est nécessaire.",
            cls.ACCURACY: 
                "Les données doivent être exactes et tenues à jour.",
            cls.STORAGE_LIMITATION: 
                "Les données doivent être conservées sous une forme permettant l'identification des personnes "
                "pour une durée n'excédant pas celle nécessaire au regard des finalités.",
            cls.INTEGRITY_CONFIDENTIALITY: 
                "Les données doivent être traitées de façon à garantir leur sécurité appropriée.",
            cls.ACCOUNTABILITY: 
                "Le responsable du traitement est responsable du respect des principes et doit être en mesure de le démontrer."
        }
        return descriptions.get(principle, "Description non disponible")


class LegalBasis(Enum):
    """Bases légales de traitement selon le RGPD (Article 6)"""
    CONSENT = auto()
    CONTRACT = auto()
    LEGAL_OBLIGATION = auto()
    VITAL_INTEREST = auto()
    PUBLIC_INTEREST = auto()
    LEGITIMATE_INTEREST = auto()
    
    @classmethod
    def get_description(cls, basis):
        """Retourne la description d'une base légale."""
        descriptions = {
            cls.CONSENT: 
                "La personne concernée a consenti au traitement de ses données à caractère personnel.",
            cls.CONTRACT: 
                "Le traitement est nécessaire à l'exécution d'un contrat auquel la personne concernée est partie.",
            cls.LEGAL_OBLIGATION: 
                "Le traitement est nécessaire au respect d'une obligation légale.",
            cls.VITAL_INTEREST: 
                "Le traitement est nécessaire à la sauvegarde des intérêts vitaux de la personne concernée.",
            cls.PUBLIC_INTEREST: 
                "Le traitement est nécessaire à l'exécution d'une mission d'intérêt public.",
            cls.LEGITIMATE_INTEREST: 
                "Le traitement est nécessaire aux fins des intérêts légitimes poursuivis par le responsable du traitement."
        }
        return descriptions.get(basis, "Description non disponible")


class DataSubjectRight(Enum):
    """Droits des personnes concernées selon le RGPD (Articles 12-23)"""
    INFORMATION = auto()
    ACCESS = auto()
    RECTIFICATION = auto()
    ERASURE = auto()
    RESTRICTION = auto()
    PORTABILITY = auto()
    OBJECT = auto()
    AUTO_DECISION = auto()
    
    @classmethod
    def get_description(cls, right):
        """Retourne la description d'un droit des personnes concernées."""
        descriptions = {
            cls.INFORMATION: 
                "Droit d'être informé de manière concise, transparente, compréhensible et aisément accessible.",
            cls.ACCESS: 
                "Droit d'obtenir la confirmation que des données sont traitées et d'y accéder.",
            cls.RECTIFICATION: 
                "Droit d'obtenir la rectification des données inexactes.",
            cls.ERASURE: 
                "Droit à l'effacement des données (droit à l'oubli).",
            cls.RESTRICTION: 
                "Droit à la limitation du traitement.",
            cls.PORTABILITY: 
                "Droit à la portabilité des données.",
            cls.OBJECT: 
                "Droit d'opposition au traitement.",
            cls.AUTO_DECISION: 
                "Droit de ne pas faire l'objet d'une décision fondée uniquement sur un traitement automatisé."
        }
        return descriptions.get(right, "Description non disponible")


class DataCategory(Enum):
    """Catégories de données personnelles avec niveau de sensibilité"""
    IDENTIFICATION = "identification"  # Noms, prénoms, identifiants
    CONTACT = "contact"  # Email, téléphone, adresse
    FINANCIAL = "financial"  # Coordonnées bancaires, revenus
    LOCATION = "location"  # Données de localisation
    BEHAVIORAL = "behavioral"  # Habitudes, préférences
    SPECIAL_RACIAL = "special_racial"  # Origine raciale ou ethnique
    SPECIAL_POLITICAL = "special_political"  # Opinions politiques
    SPECIAL_RELIGIOUS = "special_religious"  # Convictions religieuses
    SPECIAL_HEALTH = "special_health"  # Données de santé
    SPECIAL_BIOMETRIC = "special_biometric"  # Données biométriques
    
    @classmethod
    def is_special_category(cls, category):
        """Détermine si une catégorie est considérée comme sensible au sens de l'article 9."""
        return category.startswith("special_")


class ProcessingRecord:
    """Enregistrement de traitement conforme à l'article 30 du RGPD"""
    def __init__(self, process_name, purpose, controller_info, data_categories=None, legal_basis=None):
        self.record_id = str(uuid.uuid4())
        self.timestamp = datetime.datetime.now().isoformat()
        self.process_name = process_name
        self.purpose = purpose
        self.controller_info = controller_info
        self.data_categories = data_categories or []
        self.legal_basis = legal_basis
        self.data_subjects = []
        self.recipients = []
        self.transfers = []
        self.retention_period = None
        self.security_measures = []
        self.last_updated = self.timestamp
        
    def to_dict(self):
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "process_name": self.process_name,
            "purpose": self.purpose,
            "controller_info": self.controller_info,
            "data_categories": [dc.value if isinstance(dc, Enum) else dc for dc in self.data_categories],
            "legal_basis": self.legal_basis.name if isinstance(self.legal_basis, Enum) else self.legal_basis,
            "data_subjects": self.data_subjects,
            "recipients": self.recipients,
            "transfers": self.transfers,
            "retention_period": self.retention_period,
            "security_measures": self.security_measures,
            "last_updated": self.last_updated
        }


class DataSubjectRequest:
    """Gestion des demandes d'exercice de droits des personnes concernées"""
    def __init__(self, request_type, subject_id, subject_contact, request_details=None):
        self.request_id = str(uuid.uuid4())
        self.timestamp = datetime.datetime.now().isoformat()
        self.request_type = request_type  # Type de droit exercé (accès, effacement, etc.)
        self.subject_id = subject_id  # Identifiant de la personne concernée
        self.subject_contact = subject_contact  # Contact pour la réponse
        self.request_details = request_details
        self.status = "pending"  # pending, in_progress, completed, rejected
        self.response_details = None
        self.completion_date = None
        
    def to_dict(self):
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "request_type": self.request_type.name if isinstance(self.request_type, Enum) else self.request_type,
            "subject_id": self.subject_id,
            "subject_contact": self.subject_contact,
            "request_details": self.request_details,
            "status": self.status,
            "response_details": self.response_details,
            "completion_date": self.completion_date
        }


class GDPRComplianceManager(ConfigurableMixin):
    """
    Gère la conformité au RGPD (GDPR) pour XPLIA.
    
    Cette classe fournit les fonctionnalités suivantes :
    - Registre des traitements (Article 30)
    - Gestion des droits des personnes concernées
    - Documentation de la conformité aux principes du RGPD
    - Gestion des violations de données
    - Outils d'aide à l'analyse d'impact (AIPD/DPIA)
    - Génération de documentation conforme
    """
    def __init__(self, organization_name=None, dpo_contact=None):
        super().__init__()
        
        # Informations sur l'organisation
        self.organization_name = organization_name
        self.dpo_contact = dpo_contact
        
        # Initialisation des registres
        self.processing_activities = []
        self.data_subject_requests = []
        self.data_breaches = []
        self.dpia_assessments = []
        self.audit_trail = []
        
        # Configuration par défaut
        self._config.set_default('storage_path', os.path.join(os.getcwd(), 'gdpr_compliance'))
        self._config.set_default('max_log_entries', 1000)
        self._config.set_default('export_format', 'json')
        self._config.set_default('request_response_days', 30)  # Délai légal de réponse
        
        # Identifiant unique
        self.manager_id = str(uuid.uuid4())
        
        # Journaliser l'initialisation
        self._add_audit_record('initialization', {
            'organization_name': self.organization_name,
            'manager_id': self.manager_id
        })
    
    def _add_audit_record(self, action_type, details, subject_id=None):
        """Ajoute un enregistrement à l'audit trail"""
        record = {
            "record_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "action_type": action_type,
            "details": details,
            "subject_id": subject_id,
        }
        
        self.audit_trail.append(record)
        
        # Limiter la taille du journal si nécessaire
        max_entries = self._config.get('max_log_entries')
        if len(self.audit_trail) > max_entries:
            self.audit_trail = self.audit_trail[-max_entries:]
            
        return record["record_id"]
    
    def register_processing_activity(self, process_name, purpose, data_categories, legal_basis, controller_info=None):
        """
        Enregistre une activité de traitement dans le registre (Article 30).
        
        Args:
            process_name: Nom du traitement
            purpose: Finalité du traitement
            data_categories: Catégories de données traitées
            legal_basis: Base juridique du traitement
            controller_info: Information sur le responsable de traitement
            
        Returns:
            str: Identifiant de l'enregistrement créé
        """
        # Utiliser les informations de l'organisation si non spécifiées
        if not controller_info and self.organization_name:
            controller_info = {
                "name": self.organization_name,
                "dpo_contact": self.dpo_contact
            }
        
        # Créer l'enregistrement
        record = ProcessingRecord(
            process_name=process_name,
            purpose=purpose,
            controller_info=controller_info,
            data_categories=data_categories,
            legal_basis=legal_basis
        )
        
        self.processing_activities.append(record.to_dict())
        
        # Journaliser
        self._add_audit_record('processing_activity_registered', {
            'record_id': record.record_id,
            'process_name': process_name,
            'contains_special_categories': any(DataCategory.is_special_category(dc) for dc in data_categories)
        })
        
        return record.record_id
    
    def update_processing_activity(self, record_id, **updates):
        """
        Met à jour un enregistrement de traitement existant.
        
        Args:
            record_id: Identifiant de l'enregistrement à mettre à jour
            **updates: Champs à mettre à jour et leurs nouvelles valeurs
            
        Returns:
            bool: True si la mise à jour a réussi
        """
        for i, activity in enumerate(self.processing_activities):
            if activity.get("record_id") == record_id:
                # Mettre à jour les champs spécifiés
                for key, value in updates.items():
                    if key in activity:
                        self.processing_activities[i][key] = value
                
                # Mettre à jour la date de dernière modification
                self.processing_activities[i]["last_updated"] = datetime.datetime.now().isoformat()
                
                # Journaliser
                self._add_audit_record('processing_activity_updated', {
                    'record_id': record_id,
                    'updated_fields': list(updates.keys())
                })
                
                return True
        
        return False
    
    def register_data_subject_request(self, request_type, subject_id, subject_contact, request_details=None):
        """
        Enregistre une nouvelle demande d'exercice de droits d'une personne concernée.
        
        Args:
            request_type: Type de demande (accès, rectification, effacement, etc.)
            subject_id: Identifiant de la personne concernée
            subject_contact: Moyen de contact pour la réponse
            request_details: Détails supplémentaires sur la demande
            
        Returns:
            str: Identifiant de la demande
        """
        # Créer la demande
        request = DataSubjectRequest(
            request_type=request_type,
            subject_id=subject_id,
            subject_contact=subject_contact,
            request_details=request_details
        )
        
        # Ajouter au registre
        request_dict = request.to_dict()
        self.data_subject_requests.append(request_dict)
        
        # Calculer la date limite de réponse
        response_days = self._config.get('request_response_days')
        deadline = (datetime.datetime.fromisoformat(request.timestamp) + 
                   datetime.timedelta(days=response_days)).isoformat()
        
        # Journaliser
        self._add_audit_record('data_subject_request_registered', {
            'request_id': request.request_id,
            'request_type': request_type.name if isinstance(request_type, Enum) else request_type,
            'subject_id': subject_id,
            'deadline': deadline
        }, subject_id)
        
        return request.request_id
    
    def update_data_subject_request(self, request_id, status, response_details=None):
        """
        Met à jour le statut d'une demande d'exercice de droits.
        
        Args:
            request_id: Identifiant de la demande
            status: Nouveau statut ('pending', 'in_progress', 'completed', 'rejected')
            response_details: Détails de la réponse apportée
            
        Returns:
            bool: True si la mise à jour a réussi
        """
        for i, request in enumerate(self.data_subject_requests):
            if request.get("request_id") == request_id:
                # Mettre à jour le statut
                self.data_subject_requests[i]["status"] = status
                
                # Ajouter les détails de la réponse si fournis
                if response_details:
                    self.data_subject_requests[i]["response_details"] = response_details
                
                # Si terminé ou rejeté, ajouter la date de complétion
                if status in ["completed", "rejected"]:
                    self.data_subject_requests[i]["completion_date"] = datetime.datetime.now().isoformat()
                
                # Journaliser
                self._add_audit_record('data_subject_request_updated', {
                    'request_id': request_id,
                    'new_status': status
                }, request.get("subject_id"))
                
                return True
        
        return False
    
    def get_data_subject_requests(self, subject_id=None, status=None):
        """
        Récupère les demandes d'exercice de droits, avec possibilité de filtrer.
        
        Args:
            subject_id: Filtre par identifiant de personne concernée
            status: Filtre par statut de demande
            
        Returns:
            list: Liste des demandes correspondant aux critères
        """
        filtered_requests = self.data_subject_requests
        
        if subject_id:
            filtered_requests = [r for r in filtered_requests if r.get("subject_id") == subject_id]
            
        if status:
            filtered_requests = [r for r in filtered_requests if r.get("status") == status]
            
        return filtered_requests
    
    def calculate_request_overdue(self):
        """
        Calcule les demandes en retard de traitement selon le délai légal configuré.
        
        Returns:
            list: Liste des demandes en retard
        """
        response_days = self._config.get('request_response_days')
        now = datetime.datetime.now()
        overdue_requests = []
        
        for request in self.data_subject_requests:
            if request["status"] not in ["completed", "rejected"]:
                request_date = datetime.datetime.fromisoformat(request["timestamp"])
                deadline = request_date + datetime.timedelta(days=response_days)
                
                if now > deadline:
                    overdue_requests.append({
                        "request_id": request["request_id"],
                        "subject_id": request["subject_id"],
                        "request_type": request["request_type"],
                        "days_overdue": (now - deadline).days
                    })
        
        return overdue_requests
    
    def register_data_breach(self, breach_description, affected_data, estimated_subjects, notification_required=True):
        """
        Enregistre une violation de données personnelles (Article 33).
        
        Args:
            breach_description: Description de la violation
            affected_data: Catégories de données affectées
            estimated_subjects: Estimation du nombre de personnes concernées
            notification_required: Si la notification à l'autorité de contrôle est requise
            
        Returns:
            str: Identifiant de l'enregistrement de violation
        """
        breach_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        # Vérifier s'il y a des catégories de données sensibles
        has_special_categories = False
        for data_cat in affected_data:
            if isinstance(data_cat, DataCategory) and DataCategory.is_special_category(data_cat):
                has_special_categories = True
                break
            elif isinstance(data_cat, str) and data_cat.startswith("special_"):
                has_special_categories = True
                break
        
        # 72 heures pour notifier l'autorité de contrôle si nécessaire
        notification_deadline = None
        if notification_required:
            notify_date = datetime.datetime.fromisoformat(timestamp) + datetime.timedelta(hours=72)
            notification_deadline = notify_date.isoformat()
        
        breach = {
            "breach_id": breach_id,
            "timestamp": timestamp,
            "description": breach_description,
            "affected_data": [d.value if isinstance(d, Enum) else d for d in affected_data],
            "estimated_subjects": estimated_subjects,
            "has_special_categories": has_special_categories,
            "notification_required": notification_required,
            "notification_deadline": notification_deadline,
            "status": "open",
            "mitigation_actions": [],
            "authority_notified": False,
            "subjects_notified": False,
            "risk_level": "high" if has_special_categories else "medium"
        }
        
        # Ajouter au registre des violations
        self.data_breaches.append(breach)
        
        # Journaliser
        self._add_audit_record('data_breach_registered', {
            'breach_id': breach_id,
            'has_special_categories': has_special_categories,
            'risk_level': breach["risk_level"],
            'notification_required': notification_required
        })
        
        return breach_id
    
    def update_data_breach(self, breach_id, **updates):
        """
        Met à jour un enregistrement de violation de données.
        
        Args:
            breach_id: Identifiant de la violation
            **updates: Champs à mettre à jour
            
        Returns:
            bool: True si la mise à jour a réussi
        """
        for i, breach in enumerate(self.data_breaches):
            if breach.get("breach_id") == breach_id:
                # Mettre à jour les champs spécifiés
                for key, value in updates.items():
                    if key in breach:
                        self.data_breaches[i][key] = value
                
                # Journaliser
                self._add_audit_record('data_breach_updated', {
                    'breach_id': breach_id,
                    'updated_fields': list(updates.keys())
                })
                
                return True
        
        return False
    
    def add_breach_mitigation(self, breach_id, action_description, completed=False):
        """
        Ajoute une action d'atténuation pour une violation de données.
        
        Args:
            breach_id: Identifiant de la violation
            action_description: Description de l'action d'atténuation
            completed: Si l'action a déjà été réalisée
            
        Returns:
            bool: True si l'ajout a réussi
        """
        for i, breach in enumerate(self.data_breaches):
            if breach.get("breach_id") == breach_id:
                # Créer l'action
                action = {
                    "action_id": str(uuid.uuid4()),
                    "description": action_description,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "completed": completed,
                    "completion_date": datetime.datetime.now().isoformat() if completed else None
                }
                
                # Ajouter aux actions d'atténuation
                self.data_breaches[i]["mitigation_actions"].append(action)
                
                # Journaliser
                self._add_audit_record('breach_mitigation_added', {
                    'breach_id': breach_id,
                    'action_id': action["action_id"],
                    'completed': completed
                })
                
                return True
        
        return False
    
    def notify_authority(self, breach_id, authority_name, notification_details):
        """
        Enregistre la notification d'une violation à l'autorité de contrôle.
        
        Args:
            breach_id: Identifiant de la violation
            authority_name: Nom de l'autorité de contrôle notifiée
            notification_details: Détails de la notification
            
        Returns:
            bool: True si l'enregistrement a réussi
        """
        for i, breach in enumerate(self.data_breaches):
            if breach.get("breach_id") == breach_id:
                # Mettre à jour le statut de notification
                self.data_breaches[i]["authority_notified"] = True
                self.data_breaches[i]["authority_notification"] = {
                    "authority_name": authority_name,
                    "notification_date": datetime.datetime.now().isoformat(),
                    "details": notification_details
                }
                
                # Vérifier si la notification est faite dans les délais
                within_deadline = True
                if breach.get("notification_deadline"):
                    deadline = datetime.datetime.fromisoformat(breach["notification_deadline"])
                    within_deadline = datetime.datetime.now() <= deadline
                
                # Journaliser
                self._add_audit_record('authority_notified', {
                    'breach_id': breach_id,
                    'authority_name': authority_name,
                    'within_deadline': within_deadline
                })
                
                return True
        
        return False
    
    def start_dpia(self, processing_name, processing_purpose, processing_details=None):
        """
        Commence une Analyse d'Impact relative à la Protection des Données (AIPD/DPIA).
        
        Args:
            processing_name: Nom du traitement faisant l'objet de l'analyse
            processing_purpose: Finalité du traitement
            processing_details: Détails supplémentaires sur le traitement
            
        Returns:
            str: Identifiant de l'AIPD
        """
        dpia_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        dpia = {
            "dpia_id": dpia_id,
            "timestamp": timestamp,
            "processing_name": processing_name,
            "processing_purpose": processing_purpose,
            "processing_details": processing_details,
            "status": "draft",  # draft, in_progress, completed, approved
            "systematic_description": None,
            "necessity_proportionality": None,
            "risks_assessment": {},
            "measures_envisaged": [],
            "consultations": [],
            "dpo_recommendation": None,
            "conclusion": None,
            "approval_date": None,
            "approved_by": None,
            "last_updated": timestamp
        }
        
        # Ajouter au registre des AIPD
        self.dpia_assessments.append(dpia)
        
        # Journaliser
        self._add_audit_record('dpia_started', {
            'dpia_id': dpia_id,
            'processing_name': processing_name
        })
        
        return dpia_id
    
    def update_dpia(self, dpia_id, **updates):
        """
        Met à jour une AIPD existante.
        
        Args:
            dpia_id: Identifiant de l'AIPD
            **updates: Champs à mettre à jour
            
        Returns:
            bool: True si la mise à jour a réussi
        """
        for i, dpia in enumerate(self.dpia_assessments):
            if dpia.get("dpia_id") == dpia_id:
                # Mettre à jour les champs spécifiés
                for key, value in updates.items():
                    if key in dpia:
                        self.dpia_assessments[i][key] = value
                
                # Mettre à jour la date de dernière modification
                self.dpia_assessments[i]["last_updated"] = datetime.datetime.now().isoformat()
                
                # Journaliser
                self._add_audit_record('dpia_updated', {
                    'dpia_id': dpia_id,
                    'updated_fields': list(updates.keys())
                })
                
                return True
        
        return False
    
    def add_dpia_risk(self, dpia_id, risk_description, likelihood, impact, affected_rights=None):
        """
        Ajoute un risque identifié dans l'AIPD.
        
        Args:
            dpia_id: Identifiant de l'AIPD
            risk_description: Description du risque
            likelihood: Probabilité (low, medium, high)
            impact: Impact (low, medium, high)
            affected_rights: Droits des personnes concernées affectés
            
        Returns:
            str: Identifiant du risque ou None si échec
        """
        for i, dpia in enumerate(self.dpia_assessments):
            if dpia.get("dpia_id") == dpia_id:
                # Initialiser la section des risques si nécessaire
                if "risks" not in self.dpia_assessments[i]["risks_assessment"]:
                    self.dpia_assessments[i]["risks_assessment"]["risks"] = []
                
                # Déterminer le niveau de risque global
                risk_level = "medium"
                if likelihood == "high" and impact == "high":
                    risk_level = "high"
                elif likelihood == "low" and impact == "low":
                    risk_level = "low"
                
                # Créer le risque
                risk_id = str(uuid.uuid4())
                risk = {
                    "risk_id": risk_id,
                    "description": risk_description,
                    "likelihood": likelihood,
                    "impact": impact,
                    "risk_level": risk_level,
                    "affected_rights": affected_rights or [],
                    "measures": []
                }
                
                # Ajouter aux risques
                self.dpia_assessments[i]["risks_assessment"]["risks"].append(risk)
                
                # Mettre à jour la date de dernière modification
                self.dpia_assessments[i]["last_updated"] = datetime.datetime.now().isoformat()
                
                # Journaliser
                self._add_audit_record('dpia_risk_added', {
                    'dpia_id': dpia_id,
                    'risk_id': risk_id,
                    'risk_level': risk_level
                })
                
                return risk_id
        
        return None
    
    def add_dpia_measure(self, dpia_id, risk_id, measure_description, reduces_likelihood=False, reduces_impact=False):
        """
        Ajoute une mesure d'atténuation pour un risque identifié dans l'AIPD.
        
        Args:
            dpia_id: Identifiant de l'AIPD
            risk_id: Identifiant du risque
            measure_description: Description de la mesure
            reduces_likelihood: Si la mesure réduit la probabilité du risque
            reduces_impact: Si la mesure réduit l'impact du risque
            
        Returns:
            bool: True si l'ajout a réussi
        """
        for i, dpia in enumerate(self.dpia_assessments):
            if dpia.get("dpia_id") == dpia_id:
                # Vérifier si le risque existe
                if "risks" in dpia.get("risks_assessment", {}):
                    for j, risk in enumerate(dpia["risks_assessment"]["risks"]):
                        if risk.get("risk_id") == risk_id:
                            # Créer la mesure
                            measure_id = str(uuid.uuid4())
                            measure = {
                                "measure_id": measure_id,
                                "description": measure_description,
                                "reduces_likelihood": reduces_likelihood,
                                "reduces_impact": reduces_impact,
                                "implemented": False
                            }
                            
                            # Ajouter aux mesures
                            self.dpia_assessments[i]["risks_assessment"]["risks"][j]["measures"].append(measure)
                            
                            # Mettre à jour la date de dernière modification
                            self.dpia_assessments[i]["last_updated"] = datetime.datetime.now().isoformat()
                            
                            # Journaliser
                            self._add_audit_record('dpia_measure_added', {
                                'dpia_id': dpia_id,
                                'risk_id': risk_id,
                                'measure_id': measure_id
                            })
                            
                            return True
        
        return False
