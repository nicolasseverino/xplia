# Script de déploiement PyPI pour XPLIA
# ========================================

Write-Host "🚀 XPLIA - Déploiement PyPI" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan
Write-Host ""

# Étape 1: Nettoyage
Write-Host "📦 Étape 1: Nettoyage des builds précédents..." -ForegroundColor Yellow
Remove-Item -Path "dist", "build", "*.egg-info" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "✓ Nettoyage terminé" -ForegroundColor Green
Write-Host ""

# Étape 2: Tests
Write-Host "🧪 Étape 2: Exécution des tests..." -ForegroundColor Yellow
python test_import.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Tests échoués! Arrêt du déploiement." -ForegroundColor Red
    exit 1
}
Write-Host "✓ Tests réussis" -ForegroundColor Green
Write-Host ""

# Étape 3: Build
Write-Host "🔨 Étape 3: Construction du package..." -ForegroundColor Yellow
python -m pip install --upgrade build
python -m build
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Build échoué!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Build réussi" -ForegroundColor Green
Write-Host ""

# Étape 4: Vérification
Write-Host "🔍 Étape 4: Vérification du package..." -ForegroundColor Yellow
python -m pip install --upgrade twine
python -m twine check dist/*
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Vérification échouée!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Vérification réussie" -ForegroundColor Green
Write-Host ""

# Étape 5: Upload TestPyPI (optionnel)
Write-Host "📤 Étape 5: Upload vers TestPyPI (optionnel)..." -ForegroundColor Yellow
$testUpload = Read-Host "Voulez-vous uploader vers TestPyPI d'abord? (o/N)"
if ($testUpload -eq "o" -or $testUpload -eq "O") {
    python -m twine upload --repository testpypi dist/*
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Upload TestPyPI échoué!" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Upload TestPyPI réussi" -ForegroundColor Green
    Write-Host ""
    Write-Host "Test d'installation depuis TestPyPI:" -ForegroundColor Cyan
    Write-Host "pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple xplia" -ForegroundColor White
    Write-Host ""
    
    $continueToProduction = Read-Host "Continuer vers PyPI production? (o/N)"
    if ($continueToProduction -ne "o" -and $continueToProduction -ne "O") {
        Write-Host "Déploiement arrêté." -ForegroundColor Yellow
        exit 0
    }
}

# Étape 6: Upload PyPI
Write-Host "📤 Étape 6: Upload vers PyPI..." -ForegroundColor Yellow
$confirm = Read-Host "ATTENTION: Vous allez publier sur PyPI PRODUCTION. Confirmer? (o/N)"
if ($confirm -ne "o" -and $confirm -ne "O") {
    Write-Host "Déploiement annulé." -ForegroundColor Yellow
    exit 0
}

python -m twine upload dist/*
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Upload PyPI échoué!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🎉 DÉPLOIEMENT RÉUSSI!" -ForegroundColor Green
Write-Host "======================" -ForegroundColor Green
Write-Host ""
Write-Host "XPLIA 1.0.1 est maintenant disponible sur PyPI!" -ForegroundColor Cyan
Write-Host ""
Write-Host "Installation:" -ForegroundColor Yellow
Write-Host "  pip install xplia" -ForegroundColor White
Write-Host ""
Write-Host "Installation complète:" -ForegroundColor Yellow
Write-Host "  pip install xplia[full]" -ForegroundColor White
Write-Host ""
Write-Host "Prochaines étapes:" -ForegroundColor Yellow
Write-Host "  1. Vérifier la page PyPI: https://pypi.org/project/xplia/" -ForegroundColor White
Write-Host "  2. Tester l'installation: pip install xplia" -ForegroundColor White
Write-Host "  3. Créer un tag Git: git tag v1.0.1 && git push --tags" -ForegroundColor White
Write-Host "  4. Créer une release GitHub" -ForegroundColor White
Write-Host ""
