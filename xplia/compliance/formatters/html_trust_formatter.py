"""
Extension du formatter HTML avec support des métriques de confiance
===================================================================

Ce module étend le générateur de rapports HTML standard pour y intégrer
les métriques de confiance issues des modules d'évaluation de confiance XPLIA.
"""

import logging
import os
from typing import Dict, Any, Optional, List

from .html_formatter import HTMLReportGenerator
from .trust_formatter_mixin import TrustFormatterMixin
from ..report_base import ReportContent

logger = logging.getLogger(__name__)

class TrustHTMLReportGenerator(HTMLReportGenerator, TrustFormatterMixin):
    """
    Générateur de rapports HTML avec support des métriques de confiance.
    
    Cette classe étend le générateur de rapports HTML standard pour y intégrer
    les métriques de confiance issues des modules d'évaluation de confiance XPLIA.
    """
    
    def _generate_html(self, content: ReportContent) -> str:
        """
        Génère le contenu HTML du rapport avec métriques de confiance.
        
        Args:
            content: Contenu du rapport
            
        Returns:
            Contenu HTML du rapport
        """
        # Génération du HTML de base
        html = super()._generate_html(content)
        
        # Vérification de la présence d'explications avec métriques de confiance
        if not hasattr(content, "explanations") or not content.explanations:
            return html
            
        # Traitement des métriques de confiance pour chaque explication
        for i, explanation in enumerate(content.explanations):
            # Traitement des métriques de confiance
            trust_data = self._process_trust_metrics(explanation)
            
            # Si des métriques sont disponibles, les intégrer au rapport
            if trust_data["has_trust_metrics"]:
                # Chargement du template de métriques de confiance
                trust_template = self._get_trust_metrics_template()
                
                # Remplacement des variables dans le template
                trust_html = trust_template
                for key, value in trust_data.items():
                    if isinstance(value, str):
                        placeholder = f"{{{{ {key} }}}}"
                        trust_html = trust_html.replace(placeholder, value)
                
                # Traitement des sections conditionnelles
                trust_html = self._process_trust_template_sections(trust_html, trust_data)
                
                # Insertion du HTML des métriques de confiance dans le rapport
                # Recherche du point d'insertion après la section d'explication
                explanation_id = f"explanation-{i+1}"
                insertion_marker = f'<div id="{explanation_id}" class="explanation-section">'
                insertion_point = html.find(insertion_marker)
                
                if insertion_point != -1:
                    # Recherche de la fin de la section d'explication
                    section_end = html.find('</div>', insertion_point)
                    if section_end != -1:
                        # Insertion des métriques de confiance à la fin de la section d'explication
                        html = html[:section_end] + trust_html + html[section_end:]
                        logger.info(f"Métriques de confiance intégrées pour l'explication {i+1}")
                else:
                    # Si le point d'insertion n'est pas trouvé, ajout à la fin du rapport
                    insertion_marker = '</body>'
                    insertion_point = html.find(insertion_marker)
                    if insertion_point != -1:
                        html = html[:insertion_point] + trust_html + html[insertion_point:]
                        logger.info(f"Métriques de confiance ajoutées à la fin du rapport pour l'explication {i+1}")
        
        return html
    
    def _process_trust_template_sections(self, html: str, data: Dict[str, Any]) -> str:
        """
        Traite les sections conditionnelles dans le template de métriques de confiance.
        
        Args:
            html: Template HTML
            data: Données pour le template
            
        Returns:
            HTML avec les sections conditionnelles traitées
        """
        # Traitement des sections conditionnelles {% if key %}...{% endif %}
        for key, value in data.items():
            if_marker = f"{{% if {key} %}}"
            endif_marker = "{% endif %}"
            
            if if_marker in html and endif_marker in html:
                # S'il y a une valeur, garder le contenu entre les marqueurs
                if value:
                    html = html.replace(if_marker, "")
                    html = html.replace(endif_marker, "")
                # Sinon, supprimer le contenu entre les marqueurs
                else:
                    start_idx = html.find(if_marker)
                    end_idx = html.find(endif_marker, start_idx) + len(endif_marker)
                    if start_idx != -1 and end_idx != -1:
                        html = html[:start_idx] + html[end_idx:]
        
        # Traitement des boucles {% for item in items %}...{% endfor %}
        for key, value in data.items():
            if not isinstance(value, list):
                continue
                
            for_marker = f"{{% for item in {key} %}}"
            endfor_marker = "{% endfor %}"
            
            if for_marker in html and endfor_marker in html:
                start_idx = html.find(for_marker)
                end_idx = html.find(endfor_marker, start_idx)
                
                if start_idx != -1 and end_idx != -1:
                    # Extraction du contenu de la boucle
                    loop_content = html[start_idx + len(for_marker):end_idx]
                    
                    # Génération du contenu pour chaque élément de la liste
                    generated_content = ""
                    for item in value:
                        if isinstance(item, tuple) and len(item) == 2:
                            # Pour les tuples (type, value)
                            item_content = loop_content.replace("{{ item[0] }}", str(item[0]))
                            item_content = item_content.replace("{{ item[1] }}", str(item[1]))
                            generated_content += item_content
                        else:
                            # Pour les éléments simples
                            item_content = loop_content.replace("{{ item }}", str(item))
                            generated_content += item_content
                    
                    # Remplacement de la boucle par le contenu généré
                    html = html[:start_idx] + generated_content + html[end_idx + len(endfor_marker):]
        
        return html
