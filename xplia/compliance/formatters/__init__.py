"""
Générateurs de rapport par format pour XPLIA
===========================================

Ce module contient les implémentations spécifiques pour chaque format de rapport
(PDF, HTML, JSON, CSV, XML) supporté par le système de génération de rapports XPLIA.
"""

from typing import Dict, Type, Any
from ..report_base import BaseReportGenerator, ReportFormat

# Import des générateurs de rapport spécifiques à chaque format
from .pdf_formatter import PDFReportGenerator
from .html_formatter import HTMLReportGenerator
from .json_formatter import JSONReportGenerator
from .csv_formatter import CSVReportGenerator 
from .xml_formatter import XMLReportGenerator

# Mapping des formats aux classes de générateur
FORMATTER_REGISTRY: Dict[ReportFormat, Type[BaseReportGenerator]] = {
    ReportFormat.PDF: PDFReportGenerator,
    ReportFormat.HTML: HTMLReportGenerator,
    ReportFormat.JSON: JSONReportGenerator,
    ReportFormat.CSV: CSVReportGenerator,
    ReportFormat.XML: XMLReportGenerator
}

__all__ = [
    'PDFReportGenerator',
    'HTMLReportGenerator', 
    'JSONReportGenerator',
    'CSVReportGenerator',
    'XMLReportGenerator',
    'FORMATTER_REGISTRY'
]
