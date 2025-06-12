# Extension conformité santé (HIPAA) — XPLIA

Ce guide explique comment utiliser XPLIA pour la conformité HIPAA dans le secteur de la santé (États-Unis).

## 1. Journalisation des accès aux données de santé

```python
from xplia.compliance.hipaa import HIPAAComplianceManager
hipaa = HIPAAComplianceManager()
hipaa.log_access(user_id='dr_smith', patient_id='p123', action='view', status='granted')
hipaa.log_access(user_id='dr_smith', patient_id='p123', action='edit', status='denied', details='Accès non autorisé')
log = hipaa.export_access_log()
print(log)
```

## 2. Vérification de violation HIPAA

```python
violation = hipaa.check_violation(user_id='dr_smith', patient_id='p123', action='edit')
if violation:
    print("Violation HIPAA détectée !")
```

## 3. Vérification de conformité sectorielle

```python
from xplia.api import check_compliance
result = check_compliance(model, data, regulations=['hipaa'])
print(result.report())
```

## 4. Bonnes pratiques
- Toujours loguer chaque accès aux données sensibles.
- Vérifier régulièrement les violations et générer des rapports pour audit.
- Compléter la conformité HIPAA avec RGPD/AI Act pour les environnements internationaux.
