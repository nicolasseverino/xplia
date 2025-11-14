"""
XPLIA Command Line Interface

Provides command-line tools for explainability analysis, compliance checking,
and report generation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def main():
    """Main entry point for the XPLIA CLI."""
    parser = argparse.ArgumentParser(
        description="XPLIA - The Ultimate AI Explainability Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explain a model prediction
  xplia explain --model model.pkl --data input.csv --output report.html

  # Check GDPR compliance
  xplia compliance --type gdpr --model model.pkl --output compliance_report.pdf

  # Generate trust metrics
  xplia trust --model model.pkl --data test_data.csv --output trust_report.html

  # Start interactive dashboard
  xplia dashboard --model model.pkl --port 8050
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Explain command
    explain_parser = subparsers.add_parser("explain", help="Generate explanations for model predictions")
    explain_parser.add_argument("--model", required=True, help="Path to model file")
    explain_parser.add_argument("--data", required=True, help="Path to input data (CSV)")
    explain_parser.add_argument("--method", default="unified", choices=["shap", "lime", "unified", "gradient"], help="Explanation method")
    explain_parser.add_argument("--output", default="explanation_report.html", help="Output file path")
    explain_parser.add_argument("--audience", default="technical", choices=["novice", "basic", "intermediate", "advanced", "expert"], help="Target audience level")

    # Compliance command
    compliance_parser = subparsers.add_parser("compliance", help="Generate compliance reports")
    compliance_parser.add_argument("--type", required=True, choices=["gdpr", "ai-act", "hipaa", "all"], help="Compliance type")
    compliance_parser.add_argument("--model", required=True, help="Path to model file")
    compliance_parser.add_argument("--output", default="compliance_report.pdf", help="Output file path")
    compliance_parser.add_argument("--format", default="pdf", choices=["pdf", "html", "json"], help="Report format")

    # Trust command
    trust_parser = subparsers.add_parser("trust", help="Evaluate model trustworthiness")
    trust_parser.add_argument("--model", required=True, help="Path to model file")
    trust_parser.add_argument("--data", required=True, help="Path to test data (CSV)")
    trust_parser.add_argument("--output", default="trust_report.html", help="Output file path")
    trust_parser.add_argument("--include-uncertainty", action="store_true", help="Include uncertainty quantification")
    trust_parser.add_argument("--check-fairwashing", action="store_true", help="Check for fairwashing")

    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Start interactive dashboard")
    dashboard_parser.add_argument("--model", required=True, help="Path to model file")
    dashboard_parser.add_argument("--port", type=int, default=8050, help="Dashboard port")
    dashboard_parser.add_argument("--host", default="127.0.0.1", help="Dashboard host")
    dashboard_parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    try:
        if args.command == "explain":
            return handle_explain(args)
        elif args.command == "compliance":
            return handle_compliance(args)
        elif args.command == "trust":
            return handle_trust(args)
        elif args.command == "dashboard":
            return handle_dashboard(args)
        elif args.command == "version":
            return handle_version()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def handle_explain(args):
    """Handle the explain command."""
    import joblib

    print(f"Loading model from {args.model}...")
    try:
        model = joblib.load(args.model)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return 1

    print(f"Loading data from {args.data}...")
    try:
        data = pd.read_csv(args.data)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return 1

    print(f"Generating explanations using {args.method} method...")

    # Import here to avoid loading heavy dependencies if not needed
    from xplia import create_explainer
    from xplia.visualizations import ChartGenerator

    # Create explainer
    explainer = create_explainer(
        model,
        method=args.method,
        background_data=data.sample(min(100, len(data)))
    )

    # Generate explanations
    explanation = explainer.explain(data.iloc[:min(10, len(data))])

    # Adapt to audience if needed
    if args.audience != 'expert':
        from xplia.explainers.calibration import AudienceAdapter
        adapter = AudienceAdapter()
        explanation = adapter.adapt(explanation, audience=args.audience)

    # Generate report
    chart_gen = ChartGenerator()
    chart_gen.create_dashboard(
        explanation,
        output=args.output
    )

    print(f"✅ Explanation report saved to {args.output}")
    return 0


def handle_compliance(args):
    """Handle the compliance command."""
    import joblib

    print(f"Checking {args.type} compliance for model {args.model}...")

    try:
        model = joblib.load(args.model)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return 1

    print(f"Generating {args.format} report...")

    # Import here to avoid loading heavy dependencies if not needed
    from xplia.compliance import GDPRCompliance, AIActCompliance, HIPAACompliance

    reports = []

    if args.type in ('gdpr', 'all'):
        print("Generating GDPR compliance report...")
        gdpr = GDPRCompliance(model, model_metadata={
            'name': 'Model',
            'purpose': 'Automated decision-making'
        })
        dpia_report = gdpr.generate_dpia()
        reports.append(('GDPR DPIA', dpia_report))

    if args.type in ('ai-act', 'all'):
        print("Generating EU AI Act compliance report...")
        ai_act = AIActCompliance(model, usage_intent='general')
        ai_act_report = ai_act.generate_compliance_report()
        reports.append(('EU AI Act', ai_act_report))

    if args.type in ('hipaa', 'all'):
        print("Generating HIPAA compliance report...")
        hipaa = HIPAACompliance(model)
        hipaa_report = hipaa.generate_compliance_report()
        reports.append(('HIPAA', hipaa_report))

    # Export report(s)
    if len(reports) == 1:
        reports[0][1].export(args.output)
        print(f"✅ {reports[0][0]} compliance report saved to {args.output}")
    else:
        # Multiple reports - save separately
        for name, report in reports:
            filename = args.output.replace('.pdf', f'_{name.lower().replace(" ", "_")}.pdf')
            report.export(filename)
            print(f"✅ {name} report saved to {filename}")

    return 0


def handle_trust(args):
    """Handle the trust command."""
    import joblib

    print(f"Evaluating trust for model {args.model}...")

    try:
        model = joblib.load(args.model)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return 1

    print(f"Using test data from {args.data}...")
    try:
        data = pd.read_csv(args.data)
        if 'target' in data.columns:
            X = data.drop('target', axis=1)
            y = data['target']
        else:
            X = data
            y = model.predict(data)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return 1

    # Import here to avoid loading heavy dependencies if not needed
    from xplia import create_explainer
    from xplia.explainers.trust import UncertaintyQuantifier, FairwashingDetector
    from xplia.visualizations import ChartGenerator

    # Create explainer
    explainer = create_explainer(model, method='shap', background_data=X.sample(min(100, len(X))))

    uncertainty = None
    fairwashing = None

    if args.include_uncertainty:
        print("Including uncertainty quantification...")
        uq = UncertaintyQuantifier(model, explainer)
        uncertainty = uq.quantify(X.iloc[:min(100, len(X))])
        print(f"  Average total uncertainty: {uncertainty.total_uncertainty.mean():.3f}")

    if args.check_fairwashing:
        print("Checking for fairwashing...")
        detector = FairwashingDetector(model, explainer)
        fairwashing = detector.detect(X.iloc[:min(100, len(X))], y[:min(100, len(y))])
        if fairwashing.detected:
            print(f"  ⚠️  Fairwashing detected! Types: {fairwashing.fairwashing_types}")
        else:
            print(f"  ✅ No fairwashing detected")

    # Generate trust report
    print("Generating trust report...")
    explanation = explainer.explain(X.iloc[:min(10, len(X))])

    chart_gen = ChartGenerator()
    chart_gen.create_dashboard(
        explanation,
        uncertainty=uncertainty,
        fairwashing=fairwashing,
        output=args.output
    )

    print(f"✅ Trust report saved to {args.output}")
    return 0


def handle_dashboard(args):
    """Handle the dashboard command."""
    print(f"Starting XPLIA dashboard on {args.host}:{args.port}...")
    print(f"Loading model from {args.model}...")
    print("Dashboard URL: http://{}:{}".format(args.host, args.port))
    print("Press Ctrl+C to stop the dashboard")

    try:
        # Import here to avoid loading heavy dependencies if not needed
        from xplia.dashboard import create_dashboard

        app = create_dashboard(args.model)
        app.run_server(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        return 0

    return 0


def handle_version():
    """Handle the version command."""
    from xplia import __version__

    print(f"XPLIA version {__version__}")
    print("The Ultimate State-of-the-Art AI Explainability Library")
    print("\nInstalled components:")

    # Check for optional dependencies
    optional_deps = {
        "SHAP": "shap",
        "LIME": "lime",
        "TensorFlow": "tensorflow",
        "PyTorch": "torch",
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm",
        "CatBoost": "catboost",
        "FastAPI": "fastapi",
        "MLflow": "mlflow",
        "W&B": "wandb",
    }

    for name, module in optional_deps.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (not installed)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
