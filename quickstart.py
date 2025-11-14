#!/usr/bin/env python3
"""
XPLIA - Script de Démarrage Rapide
===================================

Ce script vous guide pour utiliser XPLIA selon votre cas d'usage.

Usage:
    python quickstart.py

Ou lancez directement une démo:
    python quickstart.py --demo basic
    python quickstart.py --demo finance
    python quickstart.py --demo healthcare
    python quickstart.py --demo tier1
    python quickstart.py --demo tier2
"""

import argparse
import sys
from typing import Optional


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_section(title: str):
    """Print a formatted section."""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}\n")


def check_installation():
    """Check if XPLIA and dependencies are installed."""
    print_section("🔍 Vérification de l'installation")

    try:
        import xplia
        print(f"✅ XPLIA version {xplia.__version__} installée")
    except ImportError:
        print("❌ XPLIA n'est pas installé!")
        print("\nPour installer:")
        print("  pip install xplia[full]")
        print("\nOu depuis le code source:")
        print("  pip install -e '.[full]'")
        return False

    # Check key dependencies
    deps = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib',
    }

    missing = []
    for name, import_name in deps.items():
        try:
            __import__(import_name)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} manquant")
            missing.append(name)

    if missing:
        print(f"\n⚠️  Dépendances manquantes: {', '.join(missing)}")
        print("Pour installer: pip install " + " ".join(missing))
        return False

    print("\n✅ Installation complète et prête!")
    return True


def show_menu():
    """Show interactive menu for use cases."""
    print_header("🚀 XPLIA - Guide de Démarrage Interactif")

    print("""
Sélectionnez votre cas d'usage:

📊 CAS D'USAGE PAR DOMAINE:
  1. Finance / Banque (Crédit, Fraude)
  2. Santé (Diagnostic, Traitement)
  3. E-commerce (Recommandations)
  4. Vision (Classification d'images)
  5. NLP (Analyse de texte)

🤖 CAS D'USAGE PAR TYPE DE MODÈLE:
  6. scikit-learn (RF, SVM, etc.)
  7. XGBoost / LightGBM / CatBoost
  8. PyTorch (Neural Networks)
  9. TensorFlow / Keras
 10. Custom Model

🔥 FONCTIONNALITÉS AVANCÉES:
 11. TIER 1 - Multimodal, GNN, RL, Time Series
 12. TIER 2 - Meta-Learning, Bayesian, MoE, RecSys
 13. TIER 3 - Quantum, NAS, Neural ODEs
 14. Compliance (GDPR, EU AI Act, HIPAA)
 15. Fairwashing Detection

📚 AUTRES:
 16. Documentation complète
 17. Exemples disponibles
 18. Tests d'installation
 19. Quitter

""")

    choice = input("Votre choix (1-19): ").strip()
    return choice


def show_finance_example():
    """Show finance/banking use case."""
    print_section("💰 Finance / Banque - Approbation de Crédit")

    code = """
from xplia import create_explainer
from xplia.compliance import GDPRCompliance, AIActCompliance
from xplia.explainers.trust import UncertaintyQuantifier, FairwashingDetector
import pandas as pd
from xgboost import XGBClassifier

# 1. Charger vos données
X_train = pd.read_csv('credit_data.csv')
y_train = pd.read_csv('credit_labels.csv')

# 2. Entraîner votre modèle
model = XGBClassifier()
model.fit(X_train, y_train)

# 3. Créer l'explainer
explainer = create_explainer(
    model,
    method='unified',  # Combine SHAP + LIME + Counterfactuals
    methods=['shap', 'lime', 'counterfactual'],
    background_data=X_train.sample(100)
)

# 4. Expliquer une décision
explanation = explainer.explain(X_test.iloc[0])
print("Raisons du rejet:", explanation.feature_importance)

# 5. Conformité GDPR (OBLIGATOIRE en UE!)
gdpr = GDPRCompliance(model, model_metadata={
    'name': 'Scoring Crédit',
    'purpose': 'Approbation prêts'
})
dpia_report = gdpr.generate_dpia()
dpia_report.export('gdpr_report.pdf')

# 6. Conformité EU AI Act (HIGH RISK pour crédit!)
ai_act = AIActCompliance(model, usage_intent='credit_scoring')
compliance_report = ai_act.generate_compliance_report()

# 7. Détecter le fairwashing (UNIQUE à XPLIA!)
detector = FairwashingDetector(model, explainer)
result = detector.detect(X_test, y_test)
if result.detected:
    print("⚠️  Fairwashing détecté:", result.fairwashing_types)
"""

    print("Code à utiliser:")
    print(code)

    print("\nFichiers exemple:")
    print("  - examples/loan_approval_system.py (complet)")
    print("  - examples/comprehensive_xplia_demo.py")

    print("\nPour lancer:")
    print("  python examples/loan_approval_system.py")


def show_sklearn_example():
    """Show scikit-learn example."""
    print_section("🌲 scikit-learn - Random Forest, SVM, etc.")

    code = """
from xplia import create_explainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd

# 1. Données
X, y = load_iris(return_X_y=True)
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X_df = pd.DataFrame(X, columns=feature_names)

# 2. Modèle
model = RandomForestClassifier(random_state=42)
model.fit(X_df, y)

# 3. Explainer (SHAP optimal pour arbres)
explainer = create_explainer(
    model,
    method='shap',
    background_data=X_df.sample(100)
)

# 4. Expliquer
explanation = explainer.explain(X_df[:5])

print("Importance des features:")
print(explanation.feature_importance)

print("\\nQualité de l'explication:")
print(explanation.quality_metrics)

# 5. Visualiser
from xplia.visualizations import ChartGenerator
chart_gen = ChartGenerator()

chart_gen.create_chart(
    chart_type='waterfall',
    data=explanation.feature_importance,
    title='Feature Importance',
    output='explanation.html'
)

print("\\n✅ Rapport généré: explanation.html")
"""

    print("Code à utiliser:")
    print(code)

    print("\nPour tester rapidement:")
    print("  python -c 'exec(open(\"quickstart.py\").read())'")


def show_tier1_features():
    """Show TIER 1 advanced features."""
    print_section("🔥 TIER 1 - Fonctionnalités Avancées")

    print("""
XPLIA inclut 6 modules TIER 1 avancés que AUCUNE autre bibliothèque n'a:

1. 🎨 MULTIMODAL AI (Vision-Language, Diffusion)
   - Expliquer CLIP, BLIP, GPT-4V
   - Expliquer Stable Diffusion, DALL-E
   - Analyse cross-modale

2. 🕸️  GRAPH NEURAL NETWORKS
   - GNNExplainer, SubgraphX, GraphSHAP
   - Drug discovery (molécules, toxicité)

3. 🎮 REINFORCEMENT LEARNING
   - Expliquer politiques RL
   - Q-values, trajectoires
   - Action importance

4. 🔄 ADVANCED COUNTERFACTUALS
   - Minimal, Feasible, Diverse
   - Actionable recommendations
   - Cost-aware suggestions

5. 📈 TIME SERIES
   - Temporal importance
   - Forecast explanations
   - Anomaly detection

6. 🎭 GENERATIVE MODELS
   - VAE latent space
   - GAN analysis
   - StyleGAN W-space

Fichier démo: examples/tier1_advanced_features_demo.py

Pour lancer:
  python examples/tier1_advanced_features_demo.py
""")


def show_tier2_features():
    """Show TIER 2 research features."""
    print_section("🎓 TIER 2 - Research Excellence")

    print("""
XPLIA inclut 6 modules TIER 2 de recherche avancée:

1. 🧠 META-LEARNING & FEW-SHOT
   - MAML explainer
   - Prototypical Networks
   - Support set influence

2. 🔣 NEURO-SYMBOLIC AI
   - Rule extraction
   - Logic-based explanations
   - IF-THEN rules

3. 📚 CONTINUAL LEARNING
   - Explanation evolution
   - Catastrophic forgetting detection

4. 🎲 BAYESIAN DEEP LEARNING
   - Uncertainty decomposition
   - Epistemic vs Aleatoric
   - Credible intervals

5. 🎯 MIXTURE OF EXPERTS (comme GPT-4)
   - Expert routing
   - Expert specialization
   - Gating analysis

6. 🎬 RECOMMENDER SYSTEMS
   - Collaborative filtering
   - Matrix factorization
   - Latent factors

Fichier démo: examples/tier2_tier3_advanced_demo.py

Pour lancer:
  python examples/tier2_tier3_advanced_demo.py
""")


def show_compliance_features():
    """Show compliance features."""
    print_section("🏛️  Conformité Réglementaire")

    print("""
XPLIA est la SEULE bibliothèque avec compliance intégrée!

✅ GDPR (Règlement Européen)
   - Right to Explanation (Article 13-15)
   - DPIA Generation (Article 35)
   - Audit trails
   - PDF reports pour auditeurs

✅ EU AI ACT
   - Risk category assessment (MINIMAL, LIMITED, HIGH, UNACCEPTABLE)
   - Documentation requirements
   - Conformité automatique

✅ HIPAA (Healthcare)
   - Patient data access logs
   - Audit trails médicaux
   - Privacy compliance

Code exemple:
""")

    code = """
from xplia.compliance import GDPRCompliance, AIActCompliance

# GDPR
gdpr = GDPRCompliance(model, model_metadata={
    'name': 'Model Name',
    'purpose': 'credit_scoring',
    'legal_basis': 'legitimate_interest'
})

dpia_report = gdpr.generate_dpia()
dpia_report.export('gdpr_report.pdf')
print("✅ GDPR DPIA report generated")

# EU AI Act
ai_act = AIActCompliance(model, usage_intent='credit_scoring')
risk = ai_act.assess_risk_category()
print(f"Risk category: {risk}")  # HIGH for credit scoring!

compliance_report = ai_act.generate_compliance_report()
compliance_report.export('ai_act_report.pdf')
"""

    print(code)


def show_fairwashing_detection():
    """Show fairwashing detection."""
    print_section("🔍 Détection de Fairwashing (UNIQUE!)")

    print("""
XPLIA est la SEULE bibliothèque capable de détecter le "fairwashing"!

Le fairwashing = Explications trompeuses qui cachent des biais

Types détectés:
  1. Feature masking (cache features sensibles)
  2. Importance shift (déplace l'importance)
  3. Bias hiding (cache les biais)
  4. Cherry picking (sélection biaisée)
  5. Threshold manipulation

Code:
""")

    code = """
from xplia.explainers.trust import FairwashingDetector

detector = FairwashingDetector(model, explainer)
result = detector.detect(X_test, y_test, sensitive_features=['gender', 'race'])

if result.detected:
    print("⚠️  FAIRWASHING DÉTECTÉ!")
    print(f"Types: {result.fairwashing_types}")
    print(f"Sévérité: {result.severity}")
    print(f"Recommandations: {result.recommendations}")
else:
    print("✅ Aucun fairwashing détecté")

# Générer rapport
result.export_report('fairwashing_analysis.pdf')
"""

    print(code)

    print("""
Cas d'usage critiques:
  - Finance (crédit, assurance)
  - RH (recrutement)
  - Justice (récidive)
  - Santé (allocation ressources)
""")


def show_examples():
    """Show available examples."""
    print_section("📚 Exemples Disponibles")

    print("""
XPLIA inclut 17+ exemples prêts à l'emploi:

DÉMOS COMPLÈTES:
  examples/loan_approval_system.py          - Système complet de crédit
  examples/comprehensive_xplia_demo.py      - Toutes les features de base
  examples/tier1_advanced_features_demo.py  - TIER 1 avancé
  examples/tier2_tier3_advanced_demo.py     - TIER 2+3 recherche

TRUST & COMPLIANCE:
  examples/interactive_trust_demo.py        - Évaluation de confiance
  examples/trust_pipeline_demo.py           - Pipeline complet
  examples/expert_evaluation_demo.py        - Validation expert

VISUALIZATIONS:
  examples/visualization_report_example.py  - Rapports visuels
  examples/pdf_visualization_demo.py        - Export PDF
  examples/standalone_visualization_demo.py - Charts standalone

Pour lancer un exemple:
  python examples/loan_approval_system.py
  python examples/tier1_advanced_features_demo.py

Pour lister tous les exemples:
  ls -la examples/
""")


def show_documentation():
    """Show documentation links."""
    print_section("📖 Documentation")

    print("""
DOCUMENTATION DISPONIBLE:

📚 Guides:
  README.md                      - Vue d'ensemble complète
  USAGE_GUIDE_FRANCAIS.md        - Guide d'utilisation en français
  COMPLETENESS_ANALYSIS.md       - Analyse de complétude
  ARCHITECTURE.md                - Architecture technique
  CONTRIBUTING.md                - Guide de contribution
  FAQ.md                         - Questions fréquentes

🚀 Quick Start:
  python quickstart.py           - Ce script!

📝 API Reference:
  https://xplia.readthedocs.io   - Documentation complète

🐛 Support:
  GitHub Issues: https://github.com/nicolasseverino/xplia/issues
  Email: contact@xplia.com

Pour lire un guide:
  cat USAGE_GUIDE_FRANCAIS.md
  cat COMPLETENESS_ANALYSIS.md
""")


def run_installation_test():
    """Run installation test."""
    print_section("🧪 Test d'Installation")

    try:
        print("Importation de XPLIA...")
        import xplia
        print(f"✅ XPLIA {xplia.__version__}")

        print("\nImportation des modules...")
        from xplia import create_explainer
        print("✅ create_explainer")

        from xplia.compliance import GDPRCompliance
        print("✅ GDPRCompliance")

        from xplia.explainers.trust import FairwashingDetector
        print("✅ FairwashingDetector")

        from xplia.visualizations import ChartGenerator
        print("✅ ChartGenerator")

        print("\nTest rapide...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import load_iris
        import pandas as pd

        X, y = load_iris(return_X_y=True)
        X_df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_df, y)

        explainer = create_explainer(model, method='shap', background_data=X_df[:50])
        explanation = explainer.explain(X_df[:2])

        print(f"✅ Explication générée avec succès")
        print(f"   Shape: {explanation.feature_importance.shape if hasattr(explanation.feature_importance, 'shape') else 'dict'}")

        print("\n" + "=" * 80)
        print("  ✅ INSTALLATION COMPLÈTE ET FONCTIONNELLE!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        print("\nPour réinstaller:")
        print("  pip install -e '.[full]'")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='XPLIA Quick Start Script')
    parser.add_argument(
        '--demo',
        choices=['basic', 'finance', 'healthcare', 'tier1', 'tier2', 'test'],
        help='Run a specific demo directly'
    )

    args = parser.parse_args()

    # Check installation first
    if not check_installation():
        return

    # Direct demo mode
    if args.demo == 'basic':
        show_sklearn_example()
        return
    elif args.demo == 'finance':
        show_finance_example()
        return
    elif args.demo == 'tier1':
        show_tier1_features()
        return
    elif args.demo == 'tier2':
        show_tier2_features()
        return
    elif args.demo == 'test':
        run_installation_test()
        return

    # Interactive mode
    while True:
        choice = show_menu()

        if choice == '1':
            show_finance_example()
        elif choice == '2':
            print_section("🏥 Santé - Diagnostic Médical")
            print("Voir: USAGE_GUIDE_FRANCAIS.md section 'Santé'")
        elif choice == '6':
            show_sklearn_example()
        elif choice == '11':
            show_tier1_features()
        elif choice == '12':
            show_tier2_features()
        elif choice == '14':
            show_compliance_features()
        elif choice == '15':
            show_fairwashing_detection()
        elif choice == '16':
            show_documentation()
        elif choice == '17':
            show_examples()
        elif choice == '18':
            run_installation_test()
        elif choice == '19':
            print("\n👋 Au revoir!\n")
            break
        else:
            print("\n❌ Choix invalide. Essayez à nouveau.\n")

        input("\nAppuyez sur Entrée pour continuer...")


if __name__ == '__main__':
    main()
