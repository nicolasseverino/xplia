# Tutoriel conformité XPLIA (RGPD & AI Act)

Ce guide montre comment utiliser les fonctionnalités de conformité réglementaire intégrées à XPLIA pour répondre aux exigences du RGPD (droit à l'explication) et de l'AI Act européen.

## 1. Audit trail RGPD automatique

Chaque explication générée via l'API XPLIA est automatiquement journalisée pour auditabilité RGPD :

```python
import xplia
model, _ = xplia.load_model("model.pkl")
result = xplia.explain_prediction(model, instance, user_id="alice")
# Export du journal RGPD
audit_trail = xplia.export_audit_trail()
print(audit_trail)
```

## 2. Log automatique AI Act

Chaque explication est également loguée pour la conformité AI Act :

```python
# Export du log AI Act
decision_log = xplia.export_decision_log()
print(decision_log)
```

## 3. Génération de rapports de conformité (multi-formats)

```python
# Rapport HTML
html = xplia.generate_report(format='html', output_path='rapport.html')
# Rapport Markdown
md = xplia.generate_report(format='markdown', output_path='rapport.md')
# Rapport JSON
js = xplia.generate_report(format='json', output_path='rapport.json')
# Rapport PDF
xplia.generate_report(format='pdf', output_path='rapport.pdf')
```

## 4. Vérification de conformité

```python
compliance = xplia.check_compliance(model, data, regulations=['gdpr', 'ai_act'])
print(compliance.report())
```

## 5. Bonnes pratiques
- Toujours fournir un `user_id` lors des explications pour une traçabilité optimale.
- Exporter régulièrement les logs pour archivage et audit.
- Générer des rapports après chaque cycle d’explicabilité ou pour les audits réglementaires.

## 6. Ressources complémentaires
- [Texte RGPD (CNIL)](https://www.cnil.fr/fr/reglement-europeen-protection-donnees)
- [AI Act européen (projet)](https://artificialintelligenceact.eu/)
