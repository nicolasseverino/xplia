"""
Tests unitaires et d'intégration pour la conformité RGPD et AI Act de XPLIA
"""
import pytest
from xplia.compliance.explanation_rights import GDPRComplianceManager
from xplia.compliance.ai_act import AIActComplianceManager, AIRiskCategory
from xplia.compliance.compliance_report import ComplianceReportGenerator

@pytest.fixture
def gdpr_manager():
    return GDPRComplianceManager()

@pytest.fixture
def ai_act_manager():
    return AIActComplianceManager(risk_category=AIRiskCategory.HIGH)

def test_gdpr_audit_trail(gdpr_manager):
    # Ajout d'une demande d'explication
    gdpr_manager.request_log.add('user_1', {'input': 42}, {'explanation': 'exp'})
    audit = gdpr_manager.export_audit_trail()
    assert len(audit) == 1
    assert audit[0]['user_id'] == 'user_1'

def test_ai_act_decision_log(ai_act_manager):
    # Ajout d'une décision
    ai_act_manager.log_decision({'input': 42}, 'output', {'explanation': 'exp'}, user_id='user_2')
    log = ai_act_manager.export_decision_log()
    assert len(log) == 1
    assert log[0]['user_id'] == 'user_2'
    assert log[0]['risk_category'] == AIRiskCategory.HIGH

def test_compliance_report_generation(tmp_path, gdpr_manager, ai_act_manager):
    # Génération de rapports multi-formats
    audit = [{'user_id': 'u', 'input': 1, 'explanation': 'ok'}]
    log = [{'user_id': 'u', 'input': 1, 'output': 2, 'explanation': 'ok', 'risk_category': AIRiskCategory.HIGH}]
    gen = ComplianceReportGenerator(organization="TestOrg", responsible="TestResp")
    # PDF
    pdf_path = tmp_path / "report.pdf"
    gen.generate_pdf(audit, log, str(pdf_path))
    assert pdf_path.exists()
    # HTML
    html = gen.generate_html(audit, log)
    assert "Rapport de conformité" in html
    # Markdown
    from xplia.api import _generate_markdown_report
    md = _generate_markdown_report(audit, log)
    assert "Journal des demandes" in md
    # JSON
    import json
    js = json.dumps({'audit_trail': audit, 'decision_log': log})
    assert 'audit_trail' in js
