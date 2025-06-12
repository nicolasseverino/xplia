"""
Tests unitaires pour la conformité HIPAA de XPLIA
"""
import pytest
from xplia.compliance.hipaa import HIPAAComplianceManager

def test_hipaa_access_log():
    hipaa = HIPAAComplianceManager()
    hipaa.log_access('user1', 'patient1', 'view', 'granted')
    hipaa.log_access('user1', 'patient1', 'edit', 'denied', details='Non autorisé')
    log = hipaa.export_access_log()
    assert len(log) == 2
    assert log[1]['status'] == 'denied'

def test_hipaa_violation_detection():
    hipaa = HIPAAComplianceManager()
    hipaa.log_access('user1', 'patient1', 'edit', 'denied')
    violation = hipaa.check_violation('user1', 'patient1', 'edit')
    assert violation is True
    violation2 = hipaa.check_violation('user1', 'patient1', 'view')
    assert violation2 is False
