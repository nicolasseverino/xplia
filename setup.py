from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="xplia",
    version="1.0.0",  # Production ready!
    author="Nicolas Severino",
    author_email="contact@xplia.com",
    description="XPLIA: The Ultimate State-of-the-Art AI Explainability Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nicolasseverino/xplia",
    project_urls={
        "Bug Tracker": "https://github.com/nicolasseverino/xplia/issues",
        "Documentation": "https://xplia.readthedocs.io",
        "Source Code": "https://github.com/nicolasseverino/xplia",
        "Changelog": "https://github.com/nicolasseverino/xplia/blob/main/CHANGELOG.md",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Legal Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir={"": "."},
    packages=find_packages(where=".", exclude=["tests", "tests.*", "examples", "examples.*"]),
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies (minimal for basic functionality)
        "numpy>=1.20.0,<2.0.0",
        "pandas>=1.3.0,<3.0.0",
        "scikit-learn>=1.0.0,<2.0.0",
        "matplotlib>=3.4.0,<4.0.0",
        "plotly>=5.3.0,<6.0.0",
        "joblib>=1.1.0,<2.0.0",
        "jinja2>=3.0.0,<4.0.0",
        "markupsafe>=2.0.0,<3.0.0",
        "pillow>=8.0.0,<11.0.0",
    ],
    extras_require={
        # XAI methods
        "xai": [
            "shap>=0.40.0,<1.0.0",
            "lime>=0.2.0,<1.0.0",
            "alibi>=0.7.0,<1.0.0",
            "interpret>=0.2.7,<1.0.0",
        ],
        # Deep learning frameworks
        "tensorflow": [
            "tensorflow>=2.6.0,<3.0.0",
            "transformers>=4.11.0,<5.0.0",
        ],
        "pytorch": [
            "torch>=1.9.0,<3.0.0",
            "torchvision>=0.10.0,<1.0.0",
            "transformers>=4.11.0,<5.0.0",
        ],
        # Gradient boosting frameworks
        "boosting": [
            "xgboost>=1.5.0,<3.0.0",
            "lightgbm>=3.3.0,<5.0.0",
            "catboost>=1.0.0,<2.0.0",
        ],
        # Visualization and reporting
        "viz": [
            "seaborn>=0.11.0,<1.0.0",
            "dash>=2.0.0,<3.0.0",
            "reportlab>=3.6.0,<5.0.0",
            "kaleido>=0.2.0,<1.0.0",  # For static image export
        ],
        # API and web integration
        "api": [
            "flask>=2.0.0,<4.0.0",
            "fastapi>=0.68.0,<1.0.0",
            "uvicorn>=0.15.0,<1.0.0",
            "pydantic>=1.8.0,<3.0.0",
        ],
        # ML Ops integrations
        "mlops": [
            "mlflow>=1.20.0,<3.0.0",
            "wandb>=0.12.0,<1.0.0",
        ],
        # Development tools
        "dev": [
            "pytest>=6.0.0,<8.0.0",
            "pytest-cov>=2.12.0,<5.0.0",
            "pytest-xdist>=2.3.0,<4.0.0",  # Parallel testing
            "pytest-timeout>=1.4.0,<3.0.0",
            "flake8>=3.9.0,<7.0.0",
            "black>=21.5b2,<24.0.0",
            "isort>=5.9.0,<6.0.0",
            "mypy>=0.910,<2.0.0",
            "sphinx>=4.0.0,<8.0.0",
            "sphinx-rtd-theme>=0.5.0,<2.0.0",
            "pre-commit>=2.13.0,<4.0.0",
            "bandit>=1.7.0,<2.0.0",  # Security linter
            "coverage>=5.5,<8.0.0",
        ],
        # Full installation (everything)
        "full": [
            "shap>=0.40.0,<1.0.0",
            "lime>=0.2.0,<1.0.0",
            "alibi>=0.7.0,<1.0.0",
            "interpret>=0.2.7,<1.0.0",
            "tensorflow>=2.6.0,<3.0.0",
            "torch>=1.9.0,<3.0.0",
            "torchvision>=0.10.0,<1.0.0",
            "transformers>=4.11.0,<5.0.0",
            "xgboost>=1.5.0,<3.0.0",
            "lightgbm>=3.3.0,<5.0.0",
            "catboost>=1.0.0,<2.0.0",
            "seaborn>=0.11.0,<1.0.0",
            "dash>=2.0.0,<3.0.0",
            "reportlab>=3.6.0,<5.0.0",
            "kaleido>=0.2.0,<1.0.0",
            "flask>=2.0.0,<4.0.0",
            "fastapi>=0.68.0,<1.0.0",
            "uvicorn>=0.15.0,<1.0.0",
            "pydantic>=1.8.0,<3.0.0",
            "mlflow>=1.20.0,<3.0.0",
            "wandb>=0.12.0,<1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "xplia=xplia.cli:main",
        ],
    },
)
