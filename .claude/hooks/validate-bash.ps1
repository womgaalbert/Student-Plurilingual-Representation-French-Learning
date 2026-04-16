# validate-bash.ps1 — Hook pre-tool French-Learning-Perceptions ML (Windows)
param([string]$Command, [string]$File)

# Règle 1 : bloquer écriture dans data/raw/
if ($File -match "data[/\\]raw" -and $Command -notmatch "preprocess") {
    Write-Error "BLOQUÉ : data/raw/ est en lecture seule. Utiliser /project:preprocess."
    exit 1
}

# Règle 2 : vérifier consentement avant train
if ($Command -match "train") {
    $clean = "data/processed/french-learning-perceptions_clean.csv"
    if (Test-Path $clean) {
        $df = Import-Csv $clean
        $invalid = $df | Where-Object { $_.consentement -notmatch "accepte" }
        if ($invalid.Count -gt 0) {
            Write-Error "BLOQUÉ : $($invalid.Count) lignes sans consentement."
            exit 1
        }
        Write-Host "Consentement OK — $($df.Count) répondants validés."
    }
}

exit 0
