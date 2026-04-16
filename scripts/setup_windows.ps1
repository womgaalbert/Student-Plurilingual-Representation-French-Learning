# setup_windows.ps1 -- French-Learning-Perceptions-in-Plurilingual-Cameroon
# Run in PowerShell from the project root

Write-Host "=== French-Learning-Perceptions ML -- Setup ===" -ForegroundColor Cyan

# 1. Create virtual environment
python -m venv .venv
if ($LASTEXITCODE -ne 0) { Write-Error "Python not found. Install Python 3.10+"; exit 1 }

# 2. Activate and install
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# 3. French NLP model
python -m spacy download fr_core_news_sm

# 4. Create empty folders (data will be added manually)
$dirs = @("data/raw","data/processed","models/h1","models/h2",
          "models/h3","models/h4","mlruns","reports","logs","encoders")
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

Write-Host ""
Write-Host "[OK] Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Copy data_FLP.csv to data/raw/"
Write-Host "  2. python src/preprocess.py --config params.yaml"
Write-Host "  3. python src/train.py --hypothesis H1 --config params.yaml"
Write-Host "  4. python src/pipeline.py --config params.yaml    (full pipeline)"
Write-Host "  5. mlflow ui --port 5000                          (view results)"
