"""
Performance benchmarks for all explainers.

Measures execution time, memory usage, and scalability of different
explainability methods.
"""

import pytest
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import sys


pytest.importorskip("memory_profiler", reason="memory_profiler required for memory benchmarks")


class TestSHAPPerformance:
    """Benchmark SHAP explainer performance."""

    @pytest.fixture
    def small_dataset(self):
        """100 samples, 10 features."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model, X[:10]

    @pytest.fixture
    def medium_dataset(self):
        """1000 samples, 20 features."""
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        return model, X[:100]

    @pytest.fixture
    def large_dataset(self):
        """10000 samples, 50 features."""
        X, y = make_classification(n_samples=10000, n_features=50, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        return model, X[:100]

    @pytest.mark.benchmark
    def test_shap_small_dataset_time(self, small_dataset, benchmark_results={}):
        """Benchmark SHAP on small dataset."""
        model, X = small_dataset

        try:
            import shap

            start = time.time()
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            elapsed = time.time() - start

            print(f"\nSHAP Small Dataset: {elapsed:.3f}s")
            benchmark_results['shap_small'] = elapsed

            assert elapsed < 5.0  # Should complete in 5 seconds
        except ImportError:
            pytest.skip("SHAP not installed")

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_shap_medium_dataset_time(self, medium_dataset):
        """Benchmark SHAP on medium dataset."""
        model, X = medium_dataset

        try:
            import shap

            start = time.time()
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            elapsed = time.time() - start

            print(f"\nSHAP Medium Dataset: {elapsed:.3f}s")

            assert elapsed < 30.0  # Should complete in 30 seconds
        except ImportError:
            pytest.skip("SHAP not installed")

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_shap_large_dataset_time(self, large_dataset):
        """Benchmark SHAP on large dataset."""
        model, X = large_dataset

        try:
            import shap

            start = time.time()
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            elapsed = time.time() - start

            print(f"\nSHAP Large Dataset: {elapsed:.3f}s")

            assert elapsed < 120.0  # Should complete in 2 minutes
        except ImportError:
            pytest.skip("SHAP not installed")


class TestLIMEPerformance:
    """Benchmark LIME explainer performance."""

    @pytest.fixture
    def small_dataset(self):
        """100 samples, 10 features."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model, X[:10]

    @pytest.mark.benchmark
    def test_lime_small_dataset_time(self, small_dataset):
        """Benchmark LIME on small dataset."""
        model, X = small_dataset

        try:
            from lime.lime_tabular import LimeTabularExplainer

            explainer = LimeTabularExplainer(
                X,
                mode='classification',
                feature_names=[f'f{i}' for i in range(10)]
            )

            start = time.time()
            for i in range(10):
                exp = explainer.explain_instance(X[i], model.predict_proba)
            elapsed = time.time() - start

            print(f"\nLIME Small Dataset (10 instances): {elapsed:.3f}s")

            assert elapsed < 10.0  # Should complete in 10 seconds
        except ImportError:
            pytest.skip("LIME not installed")


class TestPerformanceComparison:
    """Compare performance across different methods."""

    @pytest.fixture
    def benchmark_dataset(self):
        """Standard benchmark dataset."""
        X, y = make_classification(n_samples=500, n_features=20, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        return model, X[:50], y[:50]

    @pytest.mark.benchmark
    def test_compare_explainer_speeds(self, benchmark_dataset):
        """Compare speeds of different explainers."""
        model, X, y = benchmark_dataset

        results = {}

        # Benchmark SHAP
        try:
            import shap
            start = time.time()
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            results['SHAP'] = time.time() - start
        except ImportError:
            results['SHAP'] = None

        # Benchmark LIME (just a few samples)
        try:
            from lime.lime_tabular import LimeTabularExplainer
            explainer = LimeTabularExplainer(
                X,
                mode='classification',
                feature_names=[f'f{i}' for i in range(20)]
            )
            start = time.time()
            for i in range(min(5, len(X))):
                exp = explainer.explain_instance(X[i], model.predict_proba)
            results['LIME'] = time.time() - start
        except ImportError:
            results['LIME'] = None

        # Print comparison
        print("\n\nExplainer Performance Comparison:")
        print("=" * 50)
        for method, elapsed in results.items():
            if elapsed is not None:
                print(f"{method:20s}: {elapsed:.3f}s")
            else:
                print(f"{method:20s}: Not available")


class TestMemoryUsage:
    """Benchmark memory usage of explainers."""

    @pytest.mark.benchmark
    def test_shap_memory_usage(self):
        """Benchmark SHAP memory usage."""
        try:
            import shap
            from memory_profiler import memory_usage

            X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X, y)

            def run_shap():
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X[:100])

            mem_usage = memory_usage(run_shap, interval=0.1, max_usage=True)
            print(f"\nSHAP Peak Memory: {mem_usage:.2f} MB")

        except ImportError as e:
            pytest.skip(f"Required library not installed: {e}")


class TestScalability:
    """Test how methods scale with data size."""

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_shap_scalability(self):
        """Test SHAP scalability with increasing data size."""
        try:
            import shap

            sizes = [100, 500, 1000, 2000]
            times = []

            for size in sizes:
                X, y = make_classification(n_samples=size, n_features=20, random_state=42)
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X, y)

                start = time.time()
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X[:min(50, size)])
                elapsed = time.time() - start
                times.append(elapsed)

            print("\n\nSHAP Scalability:")
            print("=" * 50)
            for size, elapsed in zip(sizes, times):
                print(f"Size {size:5d}: {elapsed:.3f}s ({elapsed/size*1000:.2f}ms per sample)")

            # Check that it scales reasonably (not exponentially)
            # Time ratio should be less than size ratio squared
            if len(times) >= 2:
                time_ratio = times[-1] / times[0]
                size_ratio = sizes[-1] / sizes[0]
                assert time_ratio < size_ratio ** 2

        except ImportError:
            pytest.skip("SHAP not installed")


class TestParallelPerformance:
    """Test parallel execution performance."""

    @pytest.mark.benchmark
    def test_parallel_vs_sequential(self):
        """Compare parallel vs sequential execution."""
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

        # Sequential
        start = time.time()
        model_seq = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        model_seq.fit(X, y)
        seq_time = time.time() - start

        # Parallel
        start = time.time()
        model_par = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_par.fit(X, y)
        par_time = time.time() - start

        print(f"\n\nParallel vs Sequential Training:")
        print("=" * 50)
        print(f"Sequential (1 core):  {seq_time:.3f}s")
        print(f"Parallel (all cores): {par_time:.3f}s")
        print(f"Speedup:              {seq_time/par_time:.2f}x")

        # Parallel should be faster (unless single core system)
        assert par_time <= seq_time * 1.5  # Allow some overhead


class TestCachePerformance:
    """Test caching impact on performance."""

    @pytest.mark.benchmark
    def test_cache_effectiveness(self):
        """Test that caching improves performance."""
        X, y = make_classification(n_samples=500, n_features=20, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        try:
            import shap

            # First run (no cache)
            start = time.time()
            explainer = shap.TreeExplainer(model)
            shap_values1 = explainer.shap_values(X[:50])
            first_time = time.time() - start

            # Second run (potentially cached internally)
            start = time.time()
            shap_values2 = explainer.shap_values(X[:50])
            second_time = time.time() - start

            print(f"\n\nCache Effectiveness:")
            print("=" * 50)
            print(f"First run:  {first_time:.3f}s")
            print(f"Second run: {second_time:.3f}s")

            # Second run should be same or faster
            assert second_time <= first_time * 1.2

        except ImportError:
            pytest.skip("SHAP not installed")


def generate_benchmark_report():
    """Generate a comprehensive benchmark report."""
    print("\n" + "=" * 70)
    print("XPLIA PERFORMANCE BENCHMARK REPORT")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")

    try:
        import shap
        print(f"SHAP version: {shap.__version__}")
    except ImportError:
        print("SHAP: Not installed")

    try:
        import lime
        print(f"LIME version: Available")
    except ImportError:
        print("LIME: Not installed")

    print("=" * 70)


if __name__ == '__main__':
    generate_benchmark_report()
    pytest.main([__file__, '-v', '-m', 'benchmark'])
