# Script de deploiement PyPI pour XPLIA
# ========================================

Write-Host "XPLIA - Deploiement PyPI" -ForegroundColor Cyan
Write-Host "============================" -ForegroundColor Cyan
Write-Host ""

# Etape 1: Nettoyage
Write-Host "Etape 1: Nettoyage des builds precedents..." -ForegroundColor Yellow
Remove-Item -Path "dist", "build", "*.egg-info" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "OK - Nettoyage termine" -ForegroundColor Green
Write-Host ""

# Etape 2: Tests
Write-Host "Etape 2: Execution des tests..." -ForegroundColor Yellow
python test_import.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERREUR - Tests echoues! Arret du deploiement." -ForegroundColor Red
    exit 1
}
Write-Host "OK - Tests reussis" -ForegroundColor Green
Write-Host ""

# Etape 3: Build
Write-Host "Etape 3: Construction du package..." -ForegroundColor Yellow
python -m pip install --upgrade build
python -m build
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERREUR - Build echoue!" -ForegroundColor Red
    exit 1
}
Write-Host "OK - Build reussi" -ForegroundColor Green
Write-Host ""

# Etape 4: Verification
Write-Host "Etape 4: Verification du package..." -ForegroundColor Yellow
python -m pip install --upgrade twine
python -m twine check dist/*
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERREUR - Verification echouee!" -ForegroundColor Red
    exit 1
}
Write-Host "OK - Verification reussie" -ForegroundColor Green
Write-Host ""

# Etape 5: Upload TestPyPI (optionnel)
Write-Host "Etape 5: Upload vers TestPyPI (optionnel)..." -ForegroundColor Yellow
$testUpload = Read-Host "Voulez-vous uploader vers TestPyPI d'abord? (o/N)"
if ($testUpload -eq "o" -or $testUpload -eq "O") {
    python -m twine upload --repository testpypi dist/*
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERREUR - Upload TestPyPI echoue!" -ForegroundColor Red
        exit 1
    }
    Write-Host "OK - Upload TestPyPI reussi" -ForegroundColor Green
    Write-Host ""
    Write-Host "Test d'installation depuis TestPyPI:" -ForegroundColor Cyan
    Write-Host "pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple xplia" -ForegroundColor White
    Write-Host ""
    
    $continueToProduction = Read-Host "Continuer vers PyPI production? (o/N)"
    if ($continueToProduction -ne "o" -and $continueToProduction -ne "O") {
        Write-Host "Deploiement arrete." -ForegroundColor Yellow
        exit 0
    }
}

# Etape 6: Upload PyPI
Write-Host "Etape 6: Upload vers PyPI..." -ForegroundColor Yellow
$confirm = Read-Host "ATTENTION: Vous allez publier sur PyPI PRODUCTION. Confirmer? (o/N)"
if ($confirm -ne "o" -and $confirm -ne "O") {
    Write-Host "Deploiement annule." -ForegroundColor Yellow
    exit 0
}

python -m twine upload dist/*
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERREUR - Upload PyPI echoue!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "DEPLOIEMENT REUSSI!" -ForegroundColor Green
Write-Host "===================" -ForegroundColor Green
Write-Host ""
Write-Host "XPLIA 1.0.1 est maintenant disponible sur PyPI!" -ForegroundColor Cyan
Write-Host ""
Write-Host "Installation:" -ForegroundColor Yellow
Write-Host "  pip install xplia" -ForegroundColor White
Write-Host ""
Write-Host "Installation complete:" -ForegroundColor Yellow
Write-Host "  pip install xplia[full]" -ForegroundColor White
Write-Host ""
Write-Host "Prochaines etapes:" -ForegroundColor Yellow
Write-Host "  1. Verifier la page PyPI: https://pypi.org/project/xplia/" -ForegroundColor White
Write-Host "  2. Tester l'installation: pip install xplia" -ForegroundColor White
Write-Host "  3. Creer un tag Git: git tag v1.0.1; git push --tags" -ForegroundColor White
Write-Host "  4. Creer une release GitHub" -ForegroundColor White
Write-Host ""
