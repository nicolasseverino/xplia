"""
Complete Loan Approval System with XPLIA

This example demonstrates a production-ready loan approval system with:
- Model training and evaluation
- Comprehensive explainability
- GDPR and AI Act compliance
- Trust evaluation (uncertainty, fairwashing detection)
- Audit trails
- Interactive dashboard
- API integration

Author: XPLIA Team
License: MIT
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# XPLIA imports
from xplia import create_explainer, set_config
from xplia.compliance.gdpr import GDPRCompliance
from xplia.compliance.ai_act import AIActCompliance
from xplia.explainers.trust.uncertainty import UncertaintyQuantifier
from xplia.explainers.trust.fairwashing import FairwashingDetector
from xplia.explainers.trust.confidence_report import ConfidenceReportGenerator
from xplia.visualizations import ChartGenerator


class LoanApprovalSystem:
    """
    Production loan approval system with full explainability and compliance.

    Features:
    - Ensemble model for robust predictions
    - Multiple explanation methods
    - GDPR/AI Act compliance
    - Uncertainty quantification
    - Fairwashing detection
    - Audit trails
    """

    def __init__(self, config=None):
        """
        Initialize the loan approval system.

        Parameters
        ----------
        config : dict, optional
            System configuration.
        """
        self.config = config or self._default_config()

        # Models
        self.model = None
        self.scaler = None

        # Explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self.unified_explainer = None

        # Compliance
        self.gdpr_compliance = None
        self.ai_act_compliance = None

        # Trust evaluators
        self.uncertainty_quantifier = None
        self.fairwashing_detector = None

        # Data
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Audit trail
        self.audit_log = []

        # Configure XPLIA
        set_config('verbosity', 'INFO')
        set_config('n_jobs', -1)
        set_config('cache_enabled', True)
        set_config('audit_trail_enabled', True)

    def _default_config(self):
        """Default system configuration."""
        return {
            'model_type': 'random_forest',
            'n_estimators': 100,
            'max_depth': 10,
            'test_size': 0.2,
            'random_state': 42,
            'enable_fairwashing_detection': True,
            'enable_uncertainty_quantification': True,
            'compliance_checks': ['gdpr', 'ai_act']
        }

    def generate_sample_data(self, n_samples=2000):
        """
        Generate sample loan application data.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        pd.DataFrame
            Generated loan application data.
        """
        np.random.seed(self.config['random_state'])

        data = pd.DataFrame({
            'age': np.random.randint(18, 70, n_samples),
            'income': np.random.lognormal(10.5, 0.5, n_samples),
            'employment_length': np.random.randint(0, 40, n_samples),
            'loan_amount': np.random.uniform(5000, 500000, n_samples),
            'credit_score': np.random.randint(300, 850, n_samples),
            'debt_to_income': np.random.uniform(0, 1.5, n_samples),
            'num_credit_lines': np.random.randint(0, 20, n_samples),
            'delinquencies': np.random.poisson(0.5, n_samples),
            'has_mortgage': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'has_dependents': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        })

        # Generate target (approved/rejected) with realistic logic
        approval_score = (
            (data['credit_score'] - 300) / 550 * 0.4 +
            np.log(data['income']) / 15 * 0.3 +
            (1 - data['debt_to_income']) * 0.2 +
            (1 - data['delinquencies'] / 5) * 0.1
        )

        # Add noise
        approval_score += np.random.normal(0, 0.1, n_samples)

        # Convert to binary
        data['approved'] = (approval_score > 0.5).astype(int)

        self.feature_names = [col for col in data.columns if col != 'approved']

        self._log_audit('data_generation', {
            'n_samples': n_samples,
            'n_features': len(self.feature_names),
            'approval_rate': data['approved'].mean()
        })

        return data

    def prepare_data(self, data):
        """
        Prepare data for modeling.

        Parameters
        ----------
        data : pd.DataFrame
            Raw loan application data.
        """
        # Separate features and target
        X = data[self.feature_names]
        y = data['approved']

        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )

        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        self._log_audit('data_preparation', {
            'train_size': len(self.X_train),
            'test_size': len(self.X_test),
            'train_approval_rate': self.y_train.mean(),
            'test_approval_rate': self.y_test.mean()
        })

        print(f"Data prepared:")
        print(f"  Train: {len(self.X_train)} samples")
        print(f"  Test: {len(self.X_test)} samples")
        print(f"  Features: {len(self.feature_names)}")

    def train_model(self):
        """Train the loan approval model."""
        print("\nTraining model...")

        if self.config['model_type'] == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                random_state=self.config['random_state'],
                n_jobs=-1
            )
        elif self.config['model_type'] == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                random_state=self.config['random_state']
            )
        else:
            raise ValueError(f"Unknown model type: {self.config['model_type']}")

        # Train
        self.model.fit(self.X_train_scaled, self.y_train)

        # Evaluate
        y_pred = self.model.predict(self.X_test_scaled)
        y_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]

        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_proba)
        }

        self._log_audit('model_training', {
            'model_type': self.config['model_type'],
            'metrics': metrics
        })

        print(f"\nModel Performance:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        return metrics

    def create_explainers(self):
        """Create explanation systems."""
        print("\nCreating explainers...")

        # SHAP explainer (best for tree models)
        self.shap_explainer = create_explainer(
            self.model,
            method='shap',
            background_data=self.X_train_scaled[:100],
            feature_names=self.feature_names
        )

        # LIME explainer (model-agnostic)
        self.lime_explainer = create_explainer(
            self.model,
            method='lime',
            training_data=self.X_train_scaled,
            feature_names=self.feature_names,
            mode='classification'
        )

        # Unified explainer (combines multiple methods)
        self.unified_explainer = create_explainer(
            self.model,
            method='unified',
            methods=['shap', 'lime', 'feature_importance'],
            background_data=self.X_train_scaled[:100],
            feature_names=self.feature_names
        )

        self._log_audit('explainer_creation', {
            'explainers': ['shap', 'lime', 'unified']
        })

        print("  ✓ SHAP explainer created")
        print("  ✓ LIME explainer created")
        print("  ✓ Unified explainer created")

    def explain_decision(self, instance_idx=0, method='unified'):
        """
        Explain a loan decision.

        Parameters
        ----------
        instance_idx : int
            Index of test instance to explain.

        method : str
            Explanation method ('shap', 'lime', 'unified').

        Returns
        -------
        ExplanationResult
            Explanation for the decision.
        """
        instance = self.X_test_scaled[instance_idx:instance_idx+1]
        prediction = self.model.predict(instance)[0]
        probability = self.model.predict_proba(instance)[0]

        print(f"\n{'='*60}")
        print(f"Explaining Loan Decision #{instance_idx}")
        print(f"{'='*60}")
        print(f"Decision: {'APPROVED' if prediction == 1 else 'REJECTED'}")
        print(f"Confidence: {probability[prediction]:.2%}")
        print(f"\nApplicant Details:")
        for i, feature in enumerate(self.feature_names):
            original_value = self.X_test.iloc[instance_idx][feature]
            print(f"  {feature}: {original_value:.2f}")

        # Get explanation
        if method == 'shap':
            explainer = self.shap_explainer
        elif method == 'lime':
            explainer = self.lime_explainer
        elif method == 'unified':
            explainer = self.unified_explainer
        else:
            raise ValueError(f"Unknown method: {method}")

        explanation = explainer.explain(instance)

        print(f"\nTop Influential Factors ({method.upper()}):")
        importance = explanation.explanation_data['feature_importance']
        for i, (feature, score) in enumerate(
            sorted(zip(self.feature_names, importance), key=lambda x: abs(x[1]), reverse=True)[:5]
        ):
            direction = "increases" if score > 0 else "decreases"
            print(f"  {i+1}. {feature}: {direction} approval ({abs(score):.4f})")

        self._log_audit('explanation', {
            'instance_idx': instance_idx,
            'method': method,
            'prediction': int(prediction),
            'probability': float(probability[prediction])
        })

        return explanation

    def setup_compliance(self):
        """Setup regulatory compliance modules."""
        print("\nSetting up compliance modules...")

        # GDPR Compliance
        if 'gdpr' in self.config['compliance_checks']:
            self.gdpr_compliance = GDPRCompliance(
                model=self.model,
                model_metadata={
                    'name': 'Loan Approval Model',
                    'version': '1.0.0',
                    'purpose': 'Automated loan application processing',
                    'data_sources': ['loan_applications.csv'],
                    'legal_basis': 'legitimate_interest',
                    'data_retention_days': 2555,  # 7 years
                    'responsible_party': 'AI Ethics Team'
                }
            )
            print("  ✓ GDPR compliance module initialized")

        # AI Act Compliance
        if 'ai_act' in self.config['compliance_checks']:
            self.ai_act_compliance = AIActCompliance(
                model=self.model,
                usage_intent='credit_scoring',  # High-risk category
                model_metadata={
                    'name': 'Loan Approval Model',
                    'description': 'ML model for loan approval decisions'
                }
            )
            risk_category = self.ai_act_compliance.assess_risk_category()
            print(f"  ✓ AI Act compliance module initialized (Risk: {risk_category})")

        self._log_audit('compliance_setup', {
            'regulations': self.config['compliance_checks']
        })

    def generate_compliance_reports(self, output_dir='compliance_reports'):
        """
        Generate regulatory compliance reports.

        Parameters
        ----------
        output_dir : str
            Directory to save reports.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\nGenerating compliance reports...")

        # GDPR Report
        if self.gdpr_compliance:
            gdpr_report = self.gdpr_compliance.generate_dpia()
            gdpr_path = output_path / 'gdpr_dpia_report.pdf'
            gdpr_report.export(str(gdpr_path), format='pdf')
            print(f"  ✓ GDPR DPIA report: {gdpr_path}")

        # AI Act Report
        if self.ai_act_compliance:
            ai_act_report = self.ai_act_compliance.generate_compliance_report()
            ai_act_path = output_path / 'ai_act_compliance_report.pdf'
            ai_act_report.export(str(ai_act_path), format='pdf')
            print(f"  ✓ AI Act compliance report: {ai_act_path}")

        self._log_audit('compliance_reports', {
            'output_dir': str(output_path)
        })

    def evaluate_trust(self):
        """Evaluate model trustworthiness."""
        print("\nEvaluating trust metrics...")

        # Uncertainty Quantification
        if self.config['enable_uncertainty_quantification']:
            self.uncertainty_quantifier = UncertaintyQuantifier(
                model=self.model,
                explainer=self.shap_explainer
            )

            uncertainty_results = self.uncertainty_quantifier.quantify(
                self.X_test_scaled[:100]
            )

            print(f"  Uncertainty Metrics:")
            print(f"    Global uncertainty: {uncertainty_results.global_uncertainty:.4f}")
            print(f"    Aleatoric uncertainty: {uncertainty_results.aleatoric_uncertainty:.4f}")
            print(f"    Epistemic uncertainty: {uncertainty_results.epistemic_uncertainty:.4f}")

        # Fairwashing Detection
        if self.config['enable_fairwashing_detection']:
            self.fairwashing_detector = FairwashingDetector(
                model=self.model,
                explainer=self.shap_explainer
            )

            fairwashing_results = self.fairwashing_detector.detect(
                self.X_test_scaled[:200],
                self.y_test[:200]
            )

            print(f"\n  Fairwashing Detection:")
            if fairwashing_results.detected:
                print(f"    ⚠️  Potential fairwashing detected!")
                print(f"    Types: {fairwashing_results.fairwashing_types}")
                print(f"    Severity: {fairwashing_results.severity}")
            else:
                print(f"    ✓ No fairwashing detected")

        # Confidence Report
        confidence_generator = ConfidenceReportGenerator(
            model=self.model,
            explainer=self.unified_explainer
        )

        confidence_report = confidence_generator.generate(
            self.X_test_scaled[:100]
        )

        print(f"\n  Confidence Metrics:")
        print(f"    Overall confidence: {confidence_report.overall_confidence:.2%}")
        print(f"    Explanation quality: {confidence_report.explanation_quality:.2%}")

        self._log_audit('trust_evaluation', {
            'uncertainty_enabled': self.config['enable_uncertainty_quantification'],
            'fairwashing_enabled': self.config['enable_fairwashing_detection']
        })

    def predict_with_explanation(self, applicant_data, return_explanation=True):
        """
        Make a prediction with explanation.

        Parameters
        ----------
        applicant_data : dict
            Loan applicant data.

        return_explanation : bool
            Whether to return explanation.

        Returns
        -------
        dict
            Prediction result with explanation.
        """
        # Convert to DataFrame
        df = pd.DataFrame([applicant_data])

        # Scale
        df_scaled = self.scaler.transform(df)

        # Predict
        prediction = self.model.predict(df_scaled)[0]
        probability = self.model.predict_proba(df_scaled)[0]

        result = {
            'decision': 'APPROVED' if prediction == 1 else 'REJECTED',
            'confidence': float(probability[prediction]),
            'timestamp': datetime.now().isoformat()
        }

        if return_explanation:
            explanation = self.unified_explainer.explain(df_scaled)
            result['explanation'] = {
                'top_factors': [
                    {
                        'feature': feature,
                        'importance': float(importance),
                        'value': float(applicant_data[feature])
                    }
                    for feature, importance in sorted(
                        zip(self.feature_names, explanation.explanation_data['feature_importance']),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:5]
                ]
            }

        self._log_audit('prediction', {
            'decision': result['decision'],
            'confidence': result['confidence']
        })

        return result

    def save_model(self, path='loan_approval_model.pkl'):
        """Save the trained model and components."""
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config,
            'audit_log': self.audit_log
        }

        joblib.dump(model_package, path)
        print(f"\nModel saved to {path}")

        self._log_audit('model_save', {'path': path})

    @classmethod
    def load_model(cls, path='loan_approval_model.pkl'):
        """Load a saved model."""
        model_package = joblib.load(path)

        system = cls(config=model_package['config'])
        system.model = model_package['model']
        system.scaler = model_package['scaler']
        system.feature_names = model_package['feature_names']
        system.audit_log = model_package['audit_log']

        print(f"\nModel loaded from {path}")
        return system

    def _log_audit(self, action, details):
        """Log audit trail."""
        self.audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        })

    def export_audit_trail(self, path='audit_trail.json'):
        """Export audit trail to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.audit_log, f, indent=2)

        print(f"\nAudit trail exported to {path}")


def main():
    """Main execution example."""
    print("="*60)
    print("XPLIA Loan Approval System - Complete Example")
    print("="*60)

    # Create system
    system = LoanApprovalSystem(config={
        'model_type': 'random_forest',
        'n_estimators': 100,
        'max_depth': 10,
        'enable_fairwashing_detection': True,
        'enable_uncertainty_quantification': True,
        'compliance_checks': ['gdpr', 'ai_act']
    })

    # Generate and prepare data
    data = system.generate_sample_data(n_samples=2000)
    system.prepare_data(data)

    # Train model
    system.train_model()

    # Create explainers
    system.create_explainers()

    # Explain sample decisions
    for i in range(3):
        system.explain_decision(instance_idx=i, method='unified')

    # Setup compliance
    system.setup_compliance()

    # Generate compliance reports
    system.generate_compliance_reports()

    # Evaluate trust
    system.evaluate_trust()

    # Example prediction with explanation
    print("\n" + "="*60)
    print("Example: New Loan Application")
    print("="*60)

    new_applicant = {
        'age': 35,
        'income': 75000,
        'employment_length': 8,
        'loan_amount': 250000,
        'credit_score': 720,
        'debt_to_income': 0.35,
        'num_credit_lines': 5,
        'delinquencies': 0,
        'has_mortgage': 1,
        'has_dependents': 1
    }

    result = system.predict_with_explanation(new_applicant)
    print(f"\nDecision: {result['decision']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nTop Influential Factors:")
    for i, factor in enumerate(result['explanation']['top_factors'], 1):
        print(f"  {i}. {factor['feature']}: {factor['value']:.2f} (importance: {factor['importance']:.4f})")

    # Save everything
    system.save_model('loan_approval_model.pkl')
    system.export_audit_trail('audit_trail.json')

    print("\n" + "="*60)
    print("✓ Complete! All reports and models saved.")
    print("="*60)


if __name__ == '__main__':
    main()
