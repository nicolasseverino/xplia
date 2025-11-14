"""
Advanced Bias Detection System.

Multi-level bias detection across data, model, explanations, and outcomes.

Author: XPLIA Team
License: MIT
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import warnings

from xplia.core.base import ExplanationResult


@dataclass
class BiasReport:
    """
    Comprehensive bias report.

    Attributes
    ----------
    bias_detected : bool
        Whether bias was detected.
    bias_level : str
        'low', 'medium', 'high'.
    bias_types : list of str
        Types of bias detected.
    bias_scores : dict
        Bias scores for each type.
    protected_attributes : list of str
        Protected attributes analyzed.
    recommendations : list of str
        Mitigation recommendations.
    metadata : dict
        Additional metadata.
    """
    bias_detected: bool
    bias_level: str
    bias_types: List[str]
    bias_scores: Dict[str, float]
    protected_attributes: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]


class DataBiasDetector:
    """
    Detect bias in training/test data.

    Analyzes data distribution for representation bias, label bias,
    and correlation with protected attributes.

    Parameters
    ----------
    protected_attributes : list of str
        Names of protected attributes (e.g., ['gender', 'race']).
    threshold : float
        Bias detection threshold.

    Examples
    --------
    >>> detector = DataBiasDetector(protected_attributes=['gender'])
    >>> bias_report = detector.detect(X, y, protected_attr_data)
    """

    def __init__(
        self,
        protected_attributes: List[str],
        threshold: float = 0.1
    ):
        self.protected_attributes = protected_attributes
        self.threshold = threshold

    def detect_representation_bias(
        self,
        protected_data: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Detect representation bias.

        Checks if protected groups are under/over-represented.

        Parameters
        ----------
        protected_data : ndarray
            Protected attribute values.

        Returns
        -------
        biased : bool
            True if representation bias detected.
        score : float
            Bias score (0 = balanced, 1 = maximum imbalance).
        """
        unique, counts = np.unique(protected_data, return_counts=True)

        if len(unique) < 2:
            return False, 0.0

        # Compute imbalance ratio
        max_count = np.max(counts)
        min_count = np.min(counts)
        imbalance_ratio = 1.0 - (min_count / max_count)

        biased = imbalance_ratio > self.threshold

        return biased, float(imbalance_ratio)

    def detect_label_bias(
        self,
        y: np.ndarray,
        protected_data: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Detect label bias.

        Checks if labels are correlated with protected attributes.

        Parameters
        ----------
        y : ndarray
            Labels.
        protected_data : ndarray
            Protected attribute values.

        Returns
        -------
        biased : bool
            True if label bias detected.
        score : float
            Correlation strength.
        """
        # Compute correlation
        if protected_data.dtype == object or len(np.unique(protected_data)) < 10:
            # Categorical protected attribute
            # Use chi-square or similar
            # For demo: simplified correlation
            unique_groups = np.unique(protected_data)

            if len(unique_groups) < 2:
                return False, 0.0

            # Compare label rates across groups
            label_rates = []
            for group in unique_groups:
                mask = protected_data == group
                if np.sum(mask) > 0:
                    label_rate = np.mean(y[mask])
                    label_rates.append(label_rate)

            if len(label_rates) < 2:
                return False, 0.0

            # Measure disparity
            max_rate = np.max(label_rates)
            min_rate = np.min(label_rates)
            disparity = max_rate - min_rate

            biased = disparity > self.threshold

            return biased, float(disparity)
        else:
            # Continuous protected attribute
            correlation = np.abs(np.corrcoef(y, protected_data)[0, 1])
            biased = correlation > self.threshold

            return biased, float(correlation)

    def detect(
        self,
        X: np.ndarray,
        y: np.ndarray,
        protected_data: Dict[str, np.ndarray]
    ) -> BiasReport:
        """
        Comprehensive data bias detection.

        Parameters
        ----------
        X : ndarray
            Features.
        y : ndarray
            Labels.
        protected_data : dict
            Dictionary mapping attribute names to values.

        Returns
        -------
        report : BiasReport
            Data bias report.
        """
        bias_types = []
        bias_scores = {}
        recommendations = []

        # Check each protected attribute
        for attr_name, attr_values in protected_data.items():
            # Representation bias
            repr_biased, repr_score = self.detect_representation_bias(attr_values)
            bias_scores[f'{attr_name}_representation'] = repr_score

            if repr_biased:
                bias_types.append(f'{attr_name}_representation_bias')
                recommendations.append(
                    f"Collect more data for underrepresented {attr_name} groups"
                )

            # Label bias
            label_biased, label_score = self.detect_label_bias(y, attr_values)
            bias_scores[f'{attr_name}_label'] = label_score

            if label_biased:
                bias_types.append(f'{attr_name}_label_bias')
                recommendations.append(
                    f"Investigate causal relationship between {attr_name} and labels"
                )

        # Overall assessment
        bias_detected = len(bias_types) > 0

        if bias_detected:
            avg_score = np.mean(list(bias_scores.values()))
            if avg_score > 0.3:
                bias_level = 'high'
            elif avg_score > 0.15:
                bias_level = 'medium'
            else:
                bias_level = 'low'
        else:
            bias_level = 'none'

        return BiasReport(
            bias_detected=bias_detected,
            bias_level=bias_level,
            bias_types=bias_types,
            bias_scores=bias_scores,
            protected_attributes=list(protected_data.keys()),
            recommendations=recommendations,
            metadata={'data_size': X.shape[0]}
        )


class ModelBiasDetector:
    """
    Detect bias in model predictions.

    Analyzes model for disparate impact, equalized odds violations,
    and demographic parity.

    Parameters
    ----------
    protected_attributes : list of str
        Protected attribute names.
    fairness_threshold : float
        Threshold for fairness metrics.

    Examples
    --------
    >>> detector = ModelBiasDetector(protected_attributes=['gender'])
    >>> bias_report = detector.detect(model, X_test, y_test, protected_test)
    """

    def __init__(
        self,
        protected_attributes: List[str],
        fairness_threshold: float = 0.8
    ):
        self.protected_attributes = protected_attributes
        self.fairness_threshold = fairness_threshold

    def compute_disparate_impact(
        self,
        y_pred: np.ndarray,
        protected_data: np.ndarray
    ) -> float:
        """
        Compute disparate impact ratio.

        DI = (positive rate for unprivileged) / (positive rate for privileged)
        Fair if 0.8 <= DI <= 1.25

        Parameters
        ----------
        y_pred : ndarray
            Predictions.
        protected_data : ndarray
            Protected attribute values.

        Returns
        -------
        disparate_impact : float
            Disparate impact ratio.
        """
        unique_groups = np.unique(protected_data)

        if len(unique_groups) < 2:
            return 1.0

        # Compute positive rates
        positive_rates = []
        for group in unique_groups:
            mask = protected_data == group
            if np.sum(mask) > 0:
                pos_rate = np.mean(y_pred[mask])
                positive_rates.append(pos_rate)

        if len(positive_rates) < 2:
            return 1.0

        # DI = min / max
        disparate_impact = np.min(positive_rates) / (np.max(positive_rates) + 1e-8)

        return float(disparate_impact)

    def compute_equalized_odds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_data: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute equalized odds violation.

        Measures difference in TPR and FPR across groups.

        Parameters
        ----------
        y_true : ndarray
            True labels.
        y_pred : ndarray
            Predictions.
        protected_data : ndarray
            Protected attribute values.

        Returns
        -------
        tpr_diff : float
            Difference in true positive rates.
        fpr_diff : float
            Difference in false positive rates.
        """
        unique_groups = np.unique(protected_data)

        if len(unique_groups) < 2:
            return 0.0, 0.0

        tprs = []
        fprs = []

        for group in unique_groups:
            mask = protected_data == group

            if np.sum(mask) == 0:
                continue

            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]

            # TPR
            true_positives = np.sum((y_true_group == 1) & (y_pred_group == 1))
            condition_positive = np.sum(y_true_group == 1)
            tpr = true_positives / (condition_positive + 1e-8)
            tprs.append(tpr)

            # FPR
            false_positives = np.sum((y_true_group == 0) & (y_pred_group == 1))
            condition_negative = np.sum(y_true_group == 0)
            fpr = false_positives / (condition_negative + 1e-8)
            fprs.append(fpr)

        if len(tprs) < 2 or len(fprs) < 2:
            return 0.0, 0.0

        tpr_diff = np.max(tprs) - np.min(tprs)
        fpr_diff = np.max(fprs) - np.min(fprs)

        return float(tpr_diff), float(fpr_diff)

    def detect(
        self,
        model: Any,
        X: np.ndarray,
        y_true: np.ndarray,
        protected_data: Dict[str, np.ndarray]
    ) -> BiasReport:
        """
        Comprehensive model bias detection.

        Parameters
        ----------
        model : object
            Trained model.
        X : ndarray
            Test features.
        y_true : ndarray
            True labels.
        protected_data : dict
            Protected attributes.

        Returns
        -------
        report : BiasReport
            Model bias report.
        """
        # Get predictions
        y_pred = model.predict(X)

        bias_types = []
        bias_scores = {}
        recommendations = []

        # Check each protected attribute
        for attr_name, attr_values in protected_data.items():
            # Disparate impact
            di = self.compute_disparate_impact(y_pred, attr_values)
            bias_scores[f'{attr_name}_disparate_impact'] = di

            if di < self.fairness_threshold:
                bias_types.append(f'{attr_name}_disparate_impact')
                recommendations.append(
                    f"Apply fairness constraints or reweighing for {attr_name}"
                )

            # Equalized odds
            tpr_diff, fpr_diff = self.compute_equalized_odds(y_true, y_pred, attr_values)
            bias_scores[f'{attr_name}_tpr_diff'] = tpr_diff
            bias_scores[f'{attr_name}_fpr_diff'] = fpr_diff

            if tpr_diff > 0.1 or fpr_diff > 0.1:
                bias_types.append(f'{attr_name}_equalized_odds_violation')
                recommendations.append(
                    f"Use post-processing to equalize error rates across {attr_name} groups"
                )

        # Overall assessment
        bias_detected = len(bias_types) > 0

        if bias_detected:
            # Assess severity
            di_values = [v for k, v in bias_scores.items() if 'disparate_impact' in k]
            worst_di = np.min(di_values) if di_values else 1.0

            if worst_di < 0.6:
                bias_level = 'high'
            elif worst_di < 0.75:
                bias_level = 'medium'
            else:
                bias_level = 'low'
        else:
            bias_level = 'none'

        return BiasReport(
            bias_detected=bias_detected,
            bias_level=bias_level,
            bias_types=bias_types,
            bias_scores=bias_scores,
            protected_attributes=list(protected_data.keys()),
            recommendations=recommendations,
            metadata={'n_predictions': len(y_pred)}
        )


class ExplanationBiasDetector:
    """
    Detect bias in explanations.

    Checks if explanations unfairly attribute importance to protected
    attributes or show different patterns across groups.

    Parameters
    ----------
    protected_attributes : list of str
        Protected attribute names.
    threshold : float
        Bias threshold.

    Examples
    --------
    >>> detector = ExplanationBiasDetector(protected_attributes=['gender'])
    >>> bias_report = detector.detect(explanations, protected_data)
    """

    def __init__(
        self,
        protected_attributes: List[str],
        threshold: float = 0.1
    ):
        self.protected_attributes = protected_attributes
        self.threshold = threshold

    def detect_protected_attribute_importance(
        self,
        explanations: List[ExplanationResult],
        protected_attr_indices: List[int]
    ) -> Tuple[bool, float]:
        """
        Check if protected attributes have high importance.

        Parameters
        ----------
        explanations : list of ExplanationResult
            Explanations to check.
        protected_attr_indices : list of int
            Indices of protected attributes in feature vector.

        Returns
        -------
        biased : bool
            True if protected attributes are important.
        score : float
            Average importance of protected attributes.
        """
        protected_importances = []

        for exp in explanations:
            if 'feature_importance' not in exp.explanation_data:
                continue

            importance = np.array(exp.explanation_data['feature_importance'])

            # Get importance of protected attributes
            protected_imp = np.abs(importance[protected_attr_indices])
            protected_importances.append(np.mean(protected_imp))

        if not protected_importances:
            return False, 0.0

        avg_protected_importance = np.mean(protected_importances)
        biased = avg_protected_importance > self.threshold

        return biased, float(avg_protected_importance)

    def detect_explanation_disparity(
        self,
        explanations: List[ExplanationResult],
        protected_data: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Check if explanations differ significantly across groups.

        Parameters
        ----------
        explanations : list of ExplanationResult
            Explanations.
        protected_data : ndarray
            Protected attribute values.

        Returns
        -------
        biased : bool
            True if explanation disparity detected.
        score : float
            Disparity score.
        """
        unique_groups = np.unique(protected_data)

        if len(unique_groups) < 2:
            return False, 0.0

        # Group explanations by protected attribute
        group_importances = {group: [] for group in unique_groups}

        for i, exp in enumerate(explanations):
            if 'feature_importance' not in exp.explanation_data:
                continue

            importance = np.array(exp.explanation_data['feature_importance'])
            group = protected_data[i]
            group_importances[group].append(importance)

        # Compute average importance per group
        avg_importances = []
        for group in unique_groups:
            if group_importances[group]:
                avg_imp = np.mean(group_importances[group], axis=0)
                avg_importances.append(avg_imp)

        if len(avg_importances) < 2:
            return False, 0.0

        # Measure disparity (L2 distance between group averages)
        disparities = []
        for i in range(len(avg_importances)):
            for j in range(i + 1, len(avg_importances)):
                dist = np.linalg.norm(avg_importances[i] - avg_importances[j])
                disparities.append(dist)

        avg_disparity = np.mean(disparities)
        biased = avg_disparity > self.threshold

        return biased, float(avg_disparity)

    def detect(
        self,
        explanations: List[ExplanationResult],
        protected_data: Dict[str, np.ndarray],
        protected_attr_indices: Optional[Dict[str, List[int]]] = None
    ) -> BiasReport:
        """
        Comprehensive explanation bias detection.

        Parameters
        ----------
        explanations : list of ExplanationResult
            Explanations to analyze.
        protected_data : dict
            Protected attributes.
        protected_attr_indices : dict, optional
            Indices of protected attributes in features.

        Returns
        -------
        report : BiasReport
            Explanation bias report.
        """
        bias_types = []
        bias_scores = {}
        recommendations = []

        protected_attr_indices = protected_attr_indices or {}

        # Check each protected attribute
        for attr_name, attr_values in protected_data.items():
            # Check if protected attribute is important
            if attr_name in protected_attr_indices:
                biased, score = self.detect_protected_attribute_importance(
                    explanations,
                    protected_attr_indices[attr_name]
                )
                bias_scores[f'{attr_name}_importance'] = score

                if biased:
                    bias_types.append(f'{attr_name}_high_importance')
                    recommendations.append(
                        f"Remove {attr_name} from model features or apply fairness constraints"
                    )

            # Check explanation disparity
            biased, score = self.detect_explanation_disparity(explanations, attr_values)
            bias_scores[f'{attr_name}_disparity'] = score

            if biased:
                bias_types.append(f'{attr_name}_explanation_disparity')
                recommendations.append(
                    f"Investigate why explanations differ across {attr_name} groups"
                )

        # Overall assessment
        bias_detected = len(bias_types) > 0

        if bias_detected:
            avg_score = np.mean(list(bias_scores.values()))
            if avg_score > 0.2:
                bias_level = 'high'
            elif avg_score > 0.1:
                bias_level = 'medium'
            else:
                bias_level = 'low'
        else:
            bias_level = 'none'

        return BiasReport(
            bias_detected=bias_detected,
            bias_level=bias_level,
            bias_types=bias_types,
            bias_scores=bias_scores,
            protected_attributes=list(protected_data.keys()),
            recommendations=recommendations,
            metadata={'n_explanations': len(explanations)}
        )


class ComprehensiveBiasAuditor:
    """
    Comprehensive bias auditing system.

    Combines data, model, and explanation bias detection into a
    unified audit report.

    Parameters
    ----------
    protected_attributes : list of str
        Protected attribute names.
    thresholds : dict, optional
        Custom thresholds for different bias types.

    Examples
    --------
    >>> auditor = ComprehensiveBiasAuditor(protected_attributes=['gender', 'race'])
    >>> audit_report = auditor.audit(
    ...     X_train, y_train, protected_train,
    ...     model, X_test, y_test, protected_test,
    ...     explanations
    ... )
    """

    def __init__(
        self,
        protected_attributes: List[str],
        thresholds: Optional[Dict[str, float]] = None
    ):
        self.protected_attributes = protected_attributes
        self.thresholds = thresholds or {}

        # Initialize detectors
        self.data_detector = DataBiasDetector(
            protected_attributes,
            threshold=self.thresholds.get('data', 0.1)
        )

        self.model_detector = ModelBiasDetector(
            protected_attributes,
            fairness_threshold=self.thresholds.get('model', 0.8)
        )

        self.explanation_detector = ExplanationBiasDetector(
            protected_attributes,
            threshold=self.thresholds.get('explanation', 0.1)
        )

    def audit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        protected_train: Dict[str, np.ndarray],
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        protected_test: Dict[str, np.ndarray],
        explanations: Optional[List[ExplanationResult]] = None,
        protected_attr_indices: Optional[Dict[str, List[int]]] = None
    ) -> Dict[str, BiasReport]:
        """
        Complete bias audit.

        Parameters
        ----------
        X_train : ndarray
            Training features.
        y_train : ndarray
            Training labels.
        protected_train : dict
            Protected attributes for training.
        model : object
            Trained model.
        X_test : ndarray
            Test features.
        y_test : ndarray
            Test labels.
        protected_test : dict
            Protected attributes for test.
        explanations : list of ExplanationResult, optional
            Explanations to audit.
        protected_attr_indices : dict, optional
            Feature indices of protected attributes.

        Returns
        -------
        audit_report : dict
            Complete audit with data, model, and explanation bias reports.
        """
        audit_report = {}

        # Data bias
        print("Auditing data bias...")
        data_report = self.data_detector.detect(X_train, y_train, protected_train)
        audit_report['data_bias'] = data_report

        # Model bias
        print("Auditing model bias...")
        model_report = self.model_detector.detect(model, X_test, y_test, protected_test)
        audit_report['model_bias'] = model_report

        # Explanation bias (if explanations provided)
        if explanations:
            print("Auditing explanation bias...")
            explanation_report = self.explanation_detector.detect(
                explanations,
                protected_test,
                protected_attr_indices
            )
            audit_report['explanation_bias'] = explanation_report

        # Overall summary
        all_biased = (
            data_report.bias_detected or
            model_report.bias_detected or
            (explanations and audit_report.get('explanation_bias', BiasReport(False, 'none', [], {}, [], [], {})).bias_detected)
        )

        audit_report['summary'] = {
            'overall_bias_detected': all_biased,
            'data_bias_detected': data_report.bias_detected,
            'model_bias_detected': model_report.bias_detected,
            'explanation_bias_detected': audit_report.get('explanation_bias', BiasReport(False, 'none', [], {}, [], [], {})).bias_detected,
            'all_recommendations': (
                data_report.recommendations +
                model_report.recommendations +
                audit_report.get('explanation_bias', BiasReport(False, 'none', [], {}, [], [], {})).recommendations
            )
        }

        return audit_report


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Advanced Bias Detection System - Example")
    print("=" * 80)

    # Generate synthetic biased dataset
    np.random.seed(42)

    n_samples = 1000
    n_features = 5

    # Protected attribute: gender (0 = female, 1 = male)
    # Biased: 70% male, 30% female
    gender = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])

    # Features
    X = np.random.randn(n_samples, n_features)

    # Biased labels: higher approval rate for males
    y = ((X[:, 0] + X[:, 1] + 0.5 * gender) > 0).astype(int)

    # Split
    split = 800
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    gender_train, gender_test = gender[:split], gender[split:]

    protected_train = {'gender': gender_train}
    protected_test = {'gender': gender_test}

    # Simple biased model
    class BiasedModel:
        def predict(self, X):
            # Simplified: uses gender in prediction
            gender_idx = 0  # Assume gender is accessible
            return (X[:, 0] + X[:, 1] > 0).astype(int)

        def predict_proba(self, X):
            pred = self.predict(X)
            return np.column_stack([1 - pred, pred])

    model = BiasedModel()

    print("\n1. DATA BIAS DETECTION")
    print("-" * 80)
    data_detector = DataBiasDetector(protected_attributes=['gender'])
    data_bias_report = data_detector.detect(X_train, y_train, protected_train)

    print(f"Bias detected: {data_bias_report.bias_detected}")
    print(f"Bias level: {data_bias_report.bias_level}")
    print(f"Bias types: {data_bias_report.bias_types}")
    print(f"Bias scores: {data_bias_report.bias_scores}")
    print(f"Recommendations:")
    for rec in data_bias_report.recommendations:
        print(f"  - {rec}")

    print("\n2. MODEL BIAS DETECTION")
    print("-" * 80)
    model_detector = ModelBiasDetector(protected_attributes=['gender'], fairness_threshold=0.8)
    model_bias_report = model_detector.detect(model, X_test, y_test, protected_test)

    print(f"Bias detected: {model_bias_report.bias_detected}")
    print(f"Bias level: {model_bias_report.bias_level}")
    print(f"Bias types: {model_bias_report.bias_types}")
    print(f"Bias scores: {model_bias_report.bias_scores}")
    print(f"Recommendations:")
    for rec in model_bias_report.recommendations:
        print(f"  - {rec}")

    print("\n3. EXPLANATION BIAS DETECTION")
    print("-" * 80)

    # Generate mock explanations
    explanations = []
    for i in range(len(X_test)):
        importance = np.array([0.3, 0.3, 0.1, 0.1, 0.2])
        # Simulate bias: different importance for different genders
        if gender_test[i] == 1:
            importance[0] += 0.2

        exp = ExplanationResult(
            method='mock_shap',
            explanation_data={'feature_importance': importance.tolist()},
            metadata={}
        )
        explanations.append(exp)

    explanation_detector = ExplanationBiasDetector(protected_attributes=['gender'])
    explanation_bias_report = explanation_detector.detect(
        explanations,
        protected_test,
        protected_attr_indices={}
    )

    print(f"Bias detected: {explanation_bias_report.bias_detected}")
    print(f"Bias level: {explanation_bias_report.bias_level}")
    print(f"Bias types: {explanation_bias_report.bias_types}")
    print(f"Bias scores: {explanation_bias_report.bias_scores}")

    print("\n4. COMPREHENSIVE BIAS AUDIT")
    print("-" * 80)
    auditor = ComprehensiveBiasAuditor(protected_attributes=['gender'])

    audit_report = auditor.audit(
        X_train, y_train, protected_train,
        model, X_test, y_test, protected_test,
        explanations
    )

    print(f"\nAudit Summary:")
    print(f"  Overall bias detected: {audit_report['summary']['overall_bias_detected']}")
    print(f"  Data bias: {audit_report['summary']['data_bias_detected']}")
    print(f"  Model bias: {audit_report['summary']['model_bias_detected']}")
    print(f"  Explanation bias: {audit_report['summary']['explanation_bias_detected']}")

    print(f"\nAll recommendations ({len(audit_report['summary']['all_recommendations'])} total):")
    for i, rec in enumerate(audit_report['summary']['all_recommendations'], 1):
        print(f"  {i}. {rec}")

    print("\n" + "=" * 80)
    print("Advanced bias detection demonstration complete!")
    print("=" * 80)
