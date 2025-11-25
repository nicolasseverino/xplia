"""
Script de test d'import minimal pour XPLIA
===========================================

Ce script teste que les composants critiques de XPLIA peuvent être importés.
"""

import sys
import traceback

def test_core_imports():
    """Teste les imports du core."""
    print("Testing core imports...")
    try:
        from xplia.core import (
            ExplainerBase,
            ExplanationResult,
            FeatureImportance,
            ModelMetadata,
            ExplanationQuality,
            ExplanationFormat,
            ModelFactory,
            ExplainerFactory,
            VisualizerFactory,
            Registry,
            ConfigManager,
            load_model,
            create_explainer
        )
        print("✓ Core imports successful")
        return True
    except Exception as e:
        print(f"✗ Core imports failed: {e}")
        traceback.print_exc()
        return False

def test_compliance_imports():
    """Teste les imports de compliance."""
    print("\nTesting compliance imports...")
    try:
        from xplia.compliance import ComplianceChecker
        print("✓ Compliance imports successful")
        return True
    except Exception as e:
        print(f"✗ Compliance imports failed: {e}")
        traceback.print_exc()
        return False

def test_utils_imports():
    """Teste les imports des utils."""
    print("\nTesting utils imports...")
    try:
        from xplia.utils import Timer, MemoryTracker
        print("✓ Utils imports successful")
        return True
    except Exception as e:
        print(f"✗ Utils imports failed: {e}")
        traceback.print_exc()
        return False

def test_factory_functionality():
    """Teste les fonctionnalités des factories."""
    print("\nTesting factory functionality...")
    try:
        from xplia.core.factory import ModelFactory, ExplainerFactory, VisualizerFactory
        
        # Test ModelFactory
        methods = ['load_model', 'create_adapter', 'detect_model_type']
        for method in methods:
            assert hasattr(ModelFactory, method), f"ModelFactory missing {method}"
        
        # Test ExplainerFactory
        methods = ['create', 'list_available_methods', 'get_recommended_method']
        for method in methods:
            assert hasattr(ExplainerFactory, method), f"ExplainerFactory missing {method}"
        
        # Test VisualizerFactory
        methods = ['create', 'list_available_charts', 'get_recommended_chart']
        for method in methods:
            assert hasattr(VisualizerFactory, method), f"VisualizerFactory missing {method}"
        
        print("✓ Factory functionality tests passed")
        return True
    except Exception as e:
        print(f"✗ Factory functionality tests failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Fonction principale."""
    print("=" * 60)
    print("XPLIA Import Test Suite")
    print("=" * 60)
    
    results = []
    results.append(("Core imports", test_core_imports()))
    results.append(("Compliance imports", test_compliance_imports()))
    results.append(("Utils imports", test_utils_imports()))
    results.append(("Factory functionality", test_factory_functionality()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All critical imports working!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
