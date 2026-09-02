# setup_databricks.ps1 — Deploy FLP ML pipeline to Databricks (Windows)
# Usage: powershell -File databricks/setup_databricks.ps1
# Prerequisites: databricks CLI configured (databricks configure --token)

$ErrorActionPreference = "Stop"

Write-Host "══════════════════════════════════════════════════════"
Write-Host "  FLP ML → Databricks Deployment"
Write-Host "══════════════════════════════════════════════════════"

# 1. Check CLI
if (-not (Get-Command databricks -ErrorAction SilentlyContinue)) {
    Write-Host "❌ databricks CLI not found. Install: pip install databricks-cli"
    exit 1
}
Write-Host "✅ databricks CLI found"

# 2. Import notebooks
Write-Host ""
Write-Host "📂 Importing notebooks to /Shared/FLP/ ..."
databricks workspace mkdirs /Shared/FLP
databricks workspace import_dir databricks/notebooks /Shared/FLP
Write-Host "✅ Notebooks imported"

# 3. Upload data to DBFS
Write-Host ""
Write-Host "📤 Uploading data to DBFS ..."
databricks fs mkdirs dbfs:/FileStore/flp
if (Test-Path "data/raw/data_FLP.csv") {
    databricks fs cp data/raw/data_FLP.csv dbfs:/FileStore/flp/data_FLP.csv --overwrite
    Write-Host "✅ data_FLP.csv uploaded to dbfs:/FileStore/flp/"
} else {
    Write-Host "⚠️  data_FLP.csv not found locally — upload manually via Databricks UI"
}

# 4. Create job
Write-Host ""
Write-Host "⚙️  Creating Databricks Workflow job ..."
$jobJson = databricks jobs create --json-file databricks/workflows/flp_pipeline_job.json
$jobId = ($jobJson | ConvertFrom-Json).job_id
Write-Host "✅ Job created with ID: $jobId"

# 5. Serving endpoint instructions
Write-Host ""
Write-Host "🚀 To create Model Serving endpoints, run after training:"
Write-Host "   databricks serving-endpoints create --json-file databricks/serving/endpoint_config.json"

Write-Host ""
Write-Host "══════════════════════════════════════════════════════"
Write-Host "  DEPLOYMENT COMPLETE"
Write-Host "══════════════════════════════════════════════════════"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Open Databricks workspace → /Shared/FLP/"
Write-Host "  2. Run 00_setup_environment (creates catalog + tables)"
Write-Host "  3. Run 01_preprocess → 02-05 train → 06_evaluate"
Write-Host "  4. Or trigger the full workflow: databricks jobs run-now --job-id $jobId"
Write-Host ""
Write-Host "To create serving endpoints (after first training):"
Write-Host "  databricks serving-endpoints create --json-file databricks/serving/endpoint_config.json"
