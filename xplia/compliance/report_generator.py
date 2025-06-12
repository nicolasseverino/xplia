"""
Module de génération de rapports de conformité réglementaire.
Permet de générer des rapports de conformité basés sur les modules GDPR et AI Act.
"""

import datetime
import json
import os
from typing import Dict, List, Optional, Union, Any

from ..formatters.base_formatter import BaseReportFormatter
from ..formatters.html_formatter import HTMLReportFormatter
from ..formatters.pdf_formatter import PDFReportFormatter
from ..formatters.json_formatter import JSONReportFormatter
from .gdpr import GDPRManager
from .ai_act import AIActComplianceManager
from ..visualizations.chart_generator import ChartGenerator


class ComplianceReportGenerator:
    """
    Générateur de rapports de conformité réglementaire intégrant GDPR et AI Act.
    """
    
    def __init__(
        self,
        gdpr_manager: Optional[GDPRManager] = None,
        ai_act_manager: Optional[AIActComplianceManager] = None,
        formatters: Optional[List[BaseReportFormatter]] = None,
        output_dir: str = "./reports",
    ):
        """
        Initialise le générateur de rapports de conformité.
        
        Args:
            gdpr_manager: Instance de GDPRManager pour les données de conformité GDPR
            ai_act_manager: Instance de AIActComplianceManager pour les données de conformité AI Act
            formatters: Liste des formateurs de rapport à utiliser
            output_dir: Répertoire de sortie pour les rapports générés
        """
        self.gdpr_manager = gdpr_manager
        self.ai_act_manager = ai_act_manager
        self.output_dir = output_dir
        
        # Formateurs par défaut si non spécifiés
        self.formatters = formatters or [
            HTMLReportFormatter(),
            PDFReportFormatter(),
            JSONReportFormatter()
        ]
        
        # Générateur de graphiques
        self.chart_generator = ChartGenerator()
        
        # Créer le répertoire de sortie s'il n'existe pas
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def generate_gdpr_compliance_report(
        self,
        organization_name: str,
        report_title: str = "Rapport de Conformité RGPD",
        language: str = "fr",
        include_processing_activities: bool = True,
        include_data_subject_requests: bool = True,
        include_breaches: bool = True,
        include_dpias: bool = True,
        include_analytics: bool = True,
        output_formats: List[str] = None,
    ) -> Dict[str, str]:
        """
        Génère un rapport de conformité GDPR/RGPD.
        
        Args:
            organization_name: Nom de l'organisation
            report_title: Titre du rapport
            language: Code de langue (fr, en, etc.)
            include_processing_activities: Inclure les activités de traitement
            include_data_subject_requests: Inclure les demandes des personnes concernées
            include_breaches: Inclure les violations de données
            include_dpias: Inclure les analyses d'impact
            include_analytics: Inclure les analyses et graphiques
            output_formats: Formats de sortie (html, pdf, json)
            
        Returns:
            Dict[str, str]: Chemins des fichiers générés par format
        """
        if not self.gdpr_manager:
            raise ValueError("Aucun gestionnaire GDPR n'a été fourni")
        
        # Construire les données du rapport
        report_data = self._build_gdpr_report_data(
            organization_name,
            include_processing_activities,
            include_data_subject_requests,
            include_breaches,
            include_dpias,
            include_analytics
        )
        
        # Ajouter les métadonnées du rapport
        timestamp = datetime.datetime.now().isoformat()
        report_data["metadata"] = {
            "title": report_title,
            "organization": organization_name,
            "language": language,
            "generated_at": timestamp,
            "report_type": "gdpr_compliance"
        }
        
        # Sélectionner les formateurs selon les formats demandés
        selected_formatters = self._select_formatters(output_formats) if output_formats else self.formatters
        
        # Générer les rapports dans chaque format
        output_files = {}
        for formatter in selected_formatters:
            output_filename = f"gdpr_compliance_report_{organization_name.replace(' ', '_').lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{formatter.extension}"
            output_path = os.path.join(self.output_dir, output_filename)
            
            formatter.generate_report(report_data, output_path, language=language)
            output_files[formatter.format_name] = output_path
            
        return output_files
    
    def _build_gdpr_report_data(
        self,
        organization_name: str,
        include_processing_activities: bool,
        include_data_subject_requests: bool,
        include_breaches: bool,
        include_dpias: bool,
        include_analytics: bool
    ) -> Dict[str, Any]:
        """
        Construit les données du rapport GDPR.
        
        Returns:
            Dict: Données du rapport
        """
        report_data = {
            "organization_name": organization_name,
            "sections": []
        }
        
        # Section d'introduction
        intro_section = {
            "title": "Introduction",
            "content": f"Ce rapport présente l'état de conformité RGPD pour {organization_name}.",
            "order": 1
        }
        report_data["sections"].append(intro_section)
        
        # Ajouter les sections selon les options
        order = 2
        
        if include_processing_activities:
            activities_section = self._build_processing_activities_section(order)
            report_data["sections"].append(activities_section)
            order += 1
            
        if include_data_subject_requests:
            requests_section = self._build_data_subject_requests_section(order)
            report_data["sections"].append(requests_section)
            order += 1
            
        if include_breaches:
            breaches_section = self._build_breaches_section(order)
            report_data["sections"].append(breaches_section)
            order += 1
            
        if include_dpias:
            dpias_section = self._build_dpias_section(order)
            report_data["sections"].append(dpias_section)
            order += 1
            
        if include_analytics and (include_processing_activities or include_data_subject_requests or include_breaches):
            analytics_section = self._build_gdpr_analytics_section(order)
            report_data["sections"].append(analytics_section)
            order += 1
        
        # Section de conclusion
        conclusion_section = {
            "title": "Conclusion",
            "content": "Ce rapport a été généré automatiquement par XPLIA. " +
                       "Il présente un aperçu de l'état de conformité RGPD actuel. " +
                       "Pour toute question ou préoccupation, veuillez contacter le DPO.",
            "order": order
        }
        report_data["sections"].append(conclusion_section)
        
        return report_data
    
    def _build_processing_activities_section(self, order: int) -> Dict[str, Any]:
        """Construit la section des activités de traitement."""
        section = {
            "title": "Registre des Activités de Traitement",
            "content": "Cette section présente les activités de traitement enregistrées.",
            "order": order,
            "subsections": []
        }
        
        # Ajouter les activités de traitement
        if self.gdpr_manager and self.gdpr_manager.processing_activities:
            for activity in self.gdpr_manager.processing_activities:
                subsection = {
                    "title": activity.get("process_name", "Activité de traitement"),
                    "content": f"Finalité: {activity.get('purpose', 'Non spécifiée')}\n" +
                               f"Base légale: {activity.get('legal_basis', 'Non spécifiée')}\n" +
                               f"Catégories de données: {', '.join(activity.get('data_categories', ['Non spécifiées']))}"
                }
                section["subsections"].append(subsection)
        else:
            section["content"] += "\n\nAucune activité de traitement n'a été enregistrée."
        
        return section
    
    def _build_data_subject_requests_section(self, order: int) -> Dict[str, Any]:
        """Construit la section des demandes d'exercice de droits."""
        section = {
            "title": "Demandes d'Exercice de Droits",
            "content": "Cette section présente les demandes d'exercice de droits reçues.",
            "order": order,
            "subsections": []
        }
        
        # Ajouter les demandes des personnes concernées
        if self.gdpr_manager and self.gdpr_manager.data_subject_requests:
            # Statistiques sur les demandes
            pending_count = sum(1 for r in self.gdpr_manager.data_subject_requests if r.get("status") == "pending")
            completed_count = sum(1 for r in self.gdpr_manager.data_subject_requests if r.get("status") == "completed")
            
            stats_subsection = {
                "title": "Statistiques des Demandes",
                "content": f"Nombre total de demandes: {len(self.gdpr_manager.data_subject_requests)}\n" +
                          f"Demandes en attente: {pending_count}\n" +
                          f"Demandes traitées: {completed_count}"
            }
            section["subsections"].append(stats_subsection)
            
            # Liste des demandes en retard
            overdue_requests = self.gdpr_manager.calculate_request_overdue()
            if overdue_requests:
                overdue_subsection = {
                    "title": "Demandes en Retard",
                    "content": f"Nombre de demandes en retard: {len(overdue_requests)}\n\n" +
                              "Liste des demandes en retard:\n" +
                              "\n".join([f"- ID: {r.get('request_id')}, Type: {r.get('request_type')}, Retard: {r.get('days_overdue')} jours" 
                                        for r in overdue_requests])
                }
                section["subsections"].append(overdue_subsection)
        else:
            section["content"] += "\n\nAucune demande d'exercice de droits n'a été enregistrée."
        
        return section
    
    def _build_breaches_section(self, order: int) -> Dict[str, Any]:
        """Construit la section des violations de données."""
        section = {
            "title": "Violations de Données",
            "content": "Cette section présente les violations de données enregistrées.",
            "order": order,
            "subsections": []
        }
        
        # Ajouter les violations de données
        if self.gdpr_manager and self.gdpr_manager.data_breaches:
            # Statistiques sur les violations
            open_count = sum(1 for b in self.gdpr_manager.data_breaches if b.get("status") == "open")
            closed_count = sum(1 for b in self.gdpr_manager.data_breaches if b.get("status") == "closed")
            high_risk_count = sum(1 for b in self.gdpr_manager.data_breaches if b.get("risk_level") == "high")
            
            stats_subsection = {
                "title": "Statistiques des Violations",
                "content": f"Nombre total de violations: {len(self.gdpr_manager.data_breaches)}\n" +
                          f"Violations ouvertes: {open_count}\n" +
                          f"Violations clôturées: {closed_count}\n" +
                          f"Violations à risque élevé: {high_risk_count}"
            }
            section["subsections"].append(stats_subsection)
            
            # Liste des violations actives
            if open_count > 0:
                open_breaches = [b for b in self.gdpr_manager.data_breaches if b.get("status") == "open"]
                open_subsection = {
                    "title": "Violations Actives",
                    "content": "Liste des violations actives:\n" +
                              "\n".join([f"- ID: {b.get('breach_id')}, Description: {b.get('description')}, " +
                                        f"Risque: {b.get('risk_level')}" 
                                        for b in open_breaches])
                }
                section["subsections"].append(open_subsection)
        else:
            section["content"] += "\n\nAucune violation de données n'a été enregistrée."
        
        return section
    
    def _build_dpias_section(self, order: int) -> Dict[str, Any]:
        """Construit la section des analyses d'impact."""
        section = {
            "title": "Analyses d'Impact (AIPD/DPIA)",
            "content": "Cette section présente les analyses d'impact relatives à la protection des données.",
            "order": order,
            "subsections": []
        }
        
        # Ajouter les AIPD
        if self.gdpr_manager and self.gdpr_manager.dpia_assessments:
            # Statistiques sur les AIPD
            draft_count = sum(1 for d in self.gdpr_manager.dpia_assessments if d.get("status") == "draft")
            in_progress_count = sum(1 for d in self.gdpr_manager.dpia_assessments if d.get("status") == "in_progress")
            completed_count = sum(1 for d in self.gdpr_manager.dpia_assessments if d.get("status") in ["completed", "approved"])
            
            stats_subsection = {
                "title": "Statistiques des AIPD",
                "content": f"Nombre total d'AIPD: {len(self.gdpr_manager.dpia_assessments)}\n" +
                          f"AIPD en brouillon: {draft_count}\n" +
                          f"AIPD en cours: {in_progress_count}\n" +
                          f"AIPD complétées: {completed_count}"
            }
            section["subsections"].append(stats_subsection)
            
            # Liste des AIPD actives
            active_dpias = [d for d in self.gdpr_manager.dpia_assessments if d.get("status") in ["draft", "in_progress"]]
            if active_dpias:
                active_subsection = {
                    "title": "AIPD Actives",
                    "content": "Liste des AIPD en cours:\n" +
                              "\n".join([f"- Traitement: {d.get('processing_name')}, " +
                                        f"Statut: {d.get('status')}, " +
                                        f"Dernière mise à jour: {d.get('last_updated', 'N/A')}" 
                                        for d in active_dpias])
                }
                section["subsections"].append(active_subsection)
        else:
            section["content"] += "\n\nAucune analyse d'impact n'a été enregistrée."
        
        return section
    
    def _build_gdpr_analytics_section(self, order: int) -> Dict[str, Any]:
        """Construit la section d'analytiques GDPR avec graphiques."""
        section = {
            "title": "Analyses et Visualisations",
            "content": "Cette section présente des visualisations des données de conformité RGPD.",
            "order": order,
            "visualizations": []
        }
        
        # N'ajouter des visualisations que si des données sont disponibles
        if self.gdpr_manager:
            # 1. Distribution des activités de traitement par base légale
            if self.gdpr_manager.processing_activities:
                legal_basis_data = {}
                for activity in self.gdpr_manager.processing_activities:
                    legal_basis = activity.get("legal_basis", "Non spécifiée")
                    legal_basis_data[legal_basis] = legal_basis_data.get(legal_basis, 0) + 1
                
                if legal_basis_data:
                    legal_basis_chart = {
                        "title": "Distribution des Activités de Traitement par Base Légale",
                        "chart_type": "pie",
                        "data": {
                            "labels": list(legal_basis_data.keys()),
                            "values": list(legal_basis_data.values())
                        }
                    }
                    section["visualizations"].append(legal_basis_chart)
            
            # 2. Statut des demandes d'exercice de droits
            if self.gdpr_manager.data_subject_requests:
                status_data = {}
                for request in self.gdpr_manager.data_subject_requests:
                    status = request.get("status", "Non spécifié")
                    status_data[status] = status_data.get(status, 0) + 1
                
                if status_data:
                    status_chart = {
                        "title": "Statut des Demandes d'Exercice de Droits",
                        "chart_type": "bar",
                        "data": {
                            "labels": list(status_data.keys()),
                            "values": list(status_data.values())
                        }
                    }
                    section["visualizations"].append(status_chart)
            
            # 3. Niveaux de risque des violations de données
            if self.gdpr_manager.data_breaches:
                risk_data = {}
                for breach in self.gdpr_manager.data_breaches:
                    risk_level = breach.get("risk_level", "Non spécifié")
                    risk_data[risk_level] = risk_data.get(risk_level, 0) + 1
                
                if risk_data:
                    risk_chart = {
                        "title": "Niveaux de Risque des Violations de Données",
                        "chart_type": "pie",
                        "data": {
                            "labels": list(risk_data.keys()),
                            "values": list(risk_data.values())
                        }
                    }
                    section["visualizations"].append(risk_chart)
        
        if not section.get("visualizations"):
            section["content"] += "\n\nAucune donnée suffisante pour générer des visualisations."
        
        return section
    
    def generate_ai_act_compliance_report(
        self,
        system_name: str,
        report_title: str = "Rapport de Conformité AI Act",
        language: str = "fr",
        include_risk_assessment: bool = True,
        include_requirements: bool = True,
        include_technical_docs: bool = True,
        include_analytics: bool = True,
        output_formats: List[str] = None,
    ) -> Dict[str, str]:
        """
        Génère un rapport de conformité AI Act.
        
        Args:
            system_name: Nom du système d'IA
            report_title: Titre du rapport
            language: Code de langue (fr, en, etc.)
            include_risk_assessment: Inclure l'évaluation des risques
            include_requirements: Inclure les exigences de l'AI Act
            include_technical_docs: Inclure la documentation technique
            include_analytics: Inclure les analyses et graphiques
            output_formats: Formats de sortie (html, pdf, json)
            
        Returns:
            Dict[str, str]: Chemins des fichiers générés par format
        """
        if not self.ai_act_manager:
            raise ValueError("Aucun gestionnaire AI Act n'a été fourni")
        
        # Construire les données du rapport
        report_data = self._build_ai_act_report_data(
            system_name,
            include_risk_assessment,
            include_requirements,
            include_technical_docs,
            include_analytics
        )
        
        # Ajouter les métadonnées du rapport
        timestamp = datetime.datetime.now().isoformat()
        report_data["metadata"] = {
            "title": report_title,
            "system_name": system_name,
            "language": language,
            "generated_at": timestamp,
            "report_type": "ai_act_compliance"
        }
        
        # Sélectionner les formateurs selon les formats demandés
        selected_formatters = self._select_formatters(output_formats) if output_formats else self.formatters
        
        # Générer les rapports dans chaque format
        output_files = {}
        for formatter in selected_formatters:
            output_filename = f"ai_act_compliance_report_{system_name.replace(' ', '_').lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{formatter.extension}"
            output_path = os.path.join(self.output_dir, output_filename)
            
            formatter.generate_report(report_data, output_path, language=language)
            output_files[formatter.format_name] = output_path
            
        return output_files
    
    def _build_ai_act_report_data(
        self,
        system_name: str,
        include_risk_assessment: bool,
        include_requirements: bool,
        include_technical_docs: bool,
        include_analytics: bool
    ) -> Dict[str, Any]:
        """
        Construit les données du rapport AI Act.
        
        Returns:
            Dict: Données du rapport
        """
        report_data = {
            "system_name": system_name,
            "sections": []
        }
        
        # Section d'introduction
        intro_section = {
            "title": "Introduction",
            "content": f"Ce rapport présente l'état de conformité avec l'AI Act pour le système {system_name}.",
            "order": 1
        }
        report_data["sections"].append(intro_section)
        
        # Ajouter les sections selon les options
        order = 2
        
        if include_risk_assessment:
            risk_section = self._build_ai_act_risk_section(order)
            report_data["sections"].append(risk_section)
            order += 1
            
        if include_requirements:
            req_section = self._build_ai_act_requirements_section(order)
            report_data["sections"].append(req_section)
            order += 1
            
        if include_technical_docs:
            tech_section = self._build_ai_act_technical_docs_section(order)
            report_data["sections"].append(tech_section)
            order += 1
            
        if include_analytics:
            analytics_section = self._build_ai_act_analytics_section(order)
            report_data["sections"].append(analytics_section)
            order += 1
        
        # Section de conclusion
        conclusion_section = {
            "title": "Conclusion",
            "content": "Ce rapport a été généré automatiquement par XPLIA. " +
                       "Il présente un aperçu de l'état de conformité avec l'AI Act actuel. " +
                       "Pour toute question ou préoccupation, veuillez contacter l'équipe responsable.",
            "order": order
        }
        report_data["sections"].append(conclusion_section)
        
        return report_data
    
    def _build_ai_act_risk_section(self, order: int) -> Dict[str, Any]:
        """Construit la section d'évaluation des risques AI Act."""
        section = {
            "title": "Évaluation des Risques",
            "content": "Cette section présente l'évaluation des risques selon l'AI Act.",
            "order": order,
            "subsections": []
        }
        
        # Ajouter les informations sur la classification de risque
        if self.ai_act_manager:
            # Classification du système
            risk_level = self.ai_act_manager.get_risk_level()
            risk_details = self.ai_act_manager.get_risk_classification_details()
            
            classification_subsection = {
                "title": "Classification du Risque",
                "content": f"Niveau de risque: {risk_level}\n\n" +
                           f"Détails de la classification:\n{risk_details}\n\n"
            }
            section["subsections"].append(classification_subsection)
            
            # Risques identifiés
            risks = self.ai_act_manager.get_identified_risks()
            if risks:
                risks_content = "Liste des risques identifiés:\n\n"
                for risk in risks:
                    risks_content += f"- {risk.get('description', 'Non spécifié')}\n" +
                                     f"  Probabilité: {risk.get('likelihood', 'Non spécifiée')}, " +
                                     f"Impact: {risk.get('impact', 'Non spécifié')}\n"
                
                risks_subsection = {
                    "title": "Risques Identifiés",
                    "content": risks_content
                }
                section["subsections"].append(risks_subsection)
                
            # Mesures d'atténuation
            mitigations = self.ai_act_manager.get_risk_mitigation_measures()
            if mitigations:
                mitigations_content = "Mesures d'atténuation mises en place:\n\n"
                for measure in mitigations:
                    mitigations_content += f"- {measure.get('description', 'Non spécifiée')}\n" +
                                           f"  Risque ciblé: {measure.get('target_risk', 'Non spécifié')}\n"
                
                mitigations_subsection = {
                    "title": "Mesures d'Atténuation",
                    "content": mitigations_content
                }
                section["subsections"].append(mitigations_subsection)
        else:
            section["content"] += "\n\nAucune évaluation de risques n'a été effectuée."
        
        return section
    
    def _build_ai_act_requirements_section(self, order: int) -> Dict[str, Any]:
        """Construit la section des exigences de l'AI Act."""
        section = {
            "title": "Exigences de l'AI Act",
            "content": "Cette section présente les exigences de l'AI Act applicables et leur état de conformité.",
            "order": order,
            "subsections": []
        }
        
        if self.ai_act_manager:
            # Récupérer les exigences applicables
            requirements = self.ai_act_manager.get_applicable_requirements()
            
            # Organiser par catégorie
            categories = {}
            for req in requirements:
                category = req.get("category", "Autres exigences")
                if category not in categories:
                    categories[category] = []
                categories[category].append(req)
            
            # Créer une sous-section par catégorie
            for category, reqs in categories.items():
                content = f"Liste des exigences pour la catégorie {category}:\n\n"
                
                # Compter le nombre d'exigences conformes
                compliant = sum(1 for r in reqs if r.get("status") == "compliant")
                partial = sum(1 for r in reqs if r.get("status") == "partial")
                non_compliant = sum(1 for r in reqs if r.get("status") == "non_compliant")
                
                content += f"Résumé: {compliant}/{len(reqs)} exigences conformes, {partial} partiellement conformes, {non_compliant} non conformes\n\n"
                
                for req in reqs:
                    status = req.get("status", "unknown")
                    status_emoji = "✅" if status == "compliant" else "⚠️" if status == "partial" else "❌"
                    content += f"{status_emoji} {req.get('id', '')}: {req.get('description', '')}\n"
                    
                    # Ajouter des détails si non conforme
                    if status != "compliant" and "details" in req:
                        content += f"   └─ {req.get('details', '')}\n"
                
                subsection = {
                    "title": f"Exigences: {category}",
                    "content": content
                }
                section["subsections"].append(subsection)
        else:
            section["content"] += "\n\nAucune exigence n'a été analysée."
        
        return section
    
    def _build_ai_act_technical_docs_section(self, order: int) -> Dict[str, Any]:
        """Construit la section de documentation technique AI Act."""
        section = {
            "title": "Documentation Technique",
            "content": "Cette section présente la documentation technique requise par l'AI Act.",
            "order": order,
            "subsections": []
        }
        
        if self.ai_act_manager:
            # Récupérer les documents techniques
            tech_docs = self.ai_act_manager.get_technical_documentation()
            
            if tech_docs:
                # Informations générales du système
                if "system_description" in tech_docs:
                    sys_desc = tech_docs["system_description"]
                    general_info = {
                        "title": "Description du Système",
                        "content": sys_desc
                    }
                    section["subsections"].append(general_info)
                
                # Architecture et conception
                if "architecture" in tech_docs:
                    arch_content = tech_docs["architecture"]
                    architecture = {
                        "title": "Architecture et Conception",
                        "content": arch_content
                    }
                    section["subsections"].append(architecture)
                
                # Détails de développement et formation
                if "development_details" in tech_docs:
                    dev_content = tech_docs["development_details"]
                    development = {
                        "title": "Détails de Développement",
                        "content": dev_content
                    }
                    section["subsections"].append(development)
                
                # Mesures de contrôle de qualité
                if "quality_control" in tech_docs:
                    qc_content = tech_docs["quality_control"]
                    quality = {
                        "title": "Contrôle de Qualité",
                        "content": qc_content
                    }
                    section["subsections"].append(quality)
                
                # Mesures de gouvernance des données
                if "data_governance" in tech_docs:
                    data_gov_content = tech_docs["data_governance"]
                    data_governance = {
                        "title": "Gouvernance des Données",
                        "content": data_gov_content
                    }
                    section["subsections"].append(data_governance)
                
                # Surveillance post-déploiement
                if "post_deployment" in tech_docs:
                    post_dep_content = tech_docs["post_deployment"]
                    post_deployment = {
                        "title": "Surveillance Post-Déploiement",
                        "content": post_dep_content
                    }
                    section["subsections"].append(post_deployment)
            else:
                section["content"] += "\n\nAucune documentation technique n'est disponible."
        else:
            section["content"] += "\n\nAucune documentation technique n'a été générée."
        
        return section

    def _build_ai_act_analytics_section(self, order: int) -> Dict[str, Any]:
        """Construit la section d'analytiques AI Act avec graphiques."""
        section = {
            "title": "Analyses et Visualisations",
            "content": "Cette section présente des visualisations des données de conformité AI Act.",
            "order": order,
            "visualizations": []
        }
        
        # N'ajouter des visualisations que si des données sont disponibles
        if self.ai_act_manager:
            requirements = self.ai_act_manager.get_applicable_requirements()
            risks = self.ai_act_manager.get_identified_risks()
            risk_mitigations = self.ai_act_manager.get_risk_mitigation_measures()
            
            # 1. Répartition des exigences par statut de conformité
            if requirements:
                status_data = {"compliant": 0, "partial": 0, "non_compliant": 0, "unknown": 0}
                for req in requirements:
                    status = req.get("status", "unknown")
                    status_data[status] = status_data.get(status, 0) + 1
                
                if sum(status_data.values()) > 0:
                    compliance_chart = {
                        "title": "Répartition des Exigences par Statut",
                        "chart_type": "pie",
                        "data": {
                            "labels": ["Conforme", "Partiellement conforme", "Non conforme", "Inconnu"],
                            "values": [status_data["compliant"], status_data["partial"], status_data["non_compliant"], status_data["unknown"]]
                        }
                    }
                    section["visualizations"].append(compliance_chart)
            
            # 2. Répartition des exigences par catégorie
            if requirements:
                category_data = {}
                for req in requirements:
                    category = req.get("category", "Autres")
                    category_data[category] = category_data.get(category, 0) + 1
                
                if category_data:
                    category_chart = {
                        "title": "Répartition des Exigences par Catégorie",
                        "chart_type": "bar",
                        "data": {
                            "labels": list(category_data.keys()),
                            "values": list(category_data.values())
                        }
                    }
                    section["visualizations"].append(category_chart)
            
            # 3. Niveau de risque global des risques identifiés
            if risks:
                risk_level_data = {"high": 0, "medium": 0, "low": 0}
                for risk in risks:
                    risk_level = risk.get("risk_level", "medium")
                    risk_level_data[risk_level] = risk_level_data.get(risk_level, 0) + 1
                
                if sum(risk_level_data.values()) > 0:
                    risk_chart = {
                        "title": "Niveaux de Risque Identifiés",
                        "chart_type": "pie",
                        "data": {
                            "labels": ["Elevé", "Moyen", "Faible"],
                            "values": [risk_level_data["high"], risk_level_data["medium"], risk_level_data["low"]]
                        }
                    }
                    section["visualizations"].append(risk_chart)
                
            # 4. Couverture des risques par des mesures d'atténuation
            if risks and risk_mitigations:
                # Calculer le pourcentage de risques couverts par des mesures
                risk_ids = {r.get("risk_id") for r in risks if "risk_id" in r}
                covered_risks = {m.get("target_risk_id") for m in risk_mitigations if "target_risk_id" in m}
                covered_count = len(risk_ids.intersection(covered_risks))
                
                if risk_ids:  # Éviter la division par zéro
                    coverage_percentage = (covered_count / len(risk_ids)) * 100
                    
                    coverage_chart = {
                        "title": "Couverture des Risques par des Mesures d'Atténuation",
                        "chart_type": "gauge",
                        "data": {
                            "value": coverage_percentage,
                            "min": 0,
                            "max": 100,
                            "threshold_low": 30,
                            "threshold_mid": 70
                        }
                    }
                    section["visualizations"].append(coverage_chart)
        
        if not section.get("visualizations"):
            section["content"] += "\n\nAucune donnée suffisante pour générer des visualisations."
        
        return section
            
    def _select_formatters(self, output_formats: List[str]) -> List[BaseReportFormatter]:
        """Sélectionne les formateurs selon les formats demandés."""
        selected = []
        format_map = {f.format_name: f for f in self.formatters}
        
        for fmt in output_formats:
            if fmt in format_map:
                selected.append(format_map[fmt])
        
        return selected if selected else self.formatters
