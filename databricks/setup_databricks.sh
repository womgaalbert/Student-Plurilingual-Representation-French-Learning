#!/bin/bash
# setup_databricks.sh — Deploy FLP ML pipeline to Databricks
# Usage: bash databricks/setup_databricks.sh
# Prerequisites: databricks CLI configured (databricks configure --token)

set -euo pipefail

echo "══════════════════════════════════════════════════════"
echo "  FLP ML → Databricks Deployment"
echo "══════════════════════════════════════════════════════"

# 1. Check CLI
if ! command -v databricks &> /dev/null; then
    echo "❌ databricks CLI not found. Install: pip install databricks-cli"
    exit 1
fi
echo "✅ databricks CLI found"

# 2. Import notebooks
echo ""
echo "📂 Importing notebooks to /Shared/FLP/ ..."
databricks workspace mkdirs /Shared/FLP
databricks workspace import_dir databricks/notebooks /Shared/FLP
echo "✅ Notebooks imported"

# 3. Upload data to DBFS
echo ""
echo "📤 Uploading data to DBFS ..."
databricks fs mkdirs dbfs:/FileStore/flp
if [ -f "data/raw/data_FLP.csv" ]; then
    databricks fs cp data/raw/data_FLP.csv dbfs:/FileStore/flp/data_FLP.csv --overwrite
    echo "✅ data_FLP.csv uploaded to dbfs:/FileStore/flp/"
else
    echo "⚠️  data_FLP.csv not found locally — upload manually via Databricks UI"
fi

# 4. Create job
echo ""
echo "⚙️  Creating Databricks Workflow job ..."
JOB_ID=$(databricks jobs create --json-file databricks/workflows/flp_pipeline_job.json | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "✅ Job created with ID: $JOB_ID"

# 5. Create serving endpoint (optional — requires models to be registered first)
echo ""
echo "🚀 To create Model Serving endpoints, run after training:"
echo "   databricks serving-endpoints create --json-file databricks/serving/endpoint_config.json"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  DEPLOYMENT COMPLETE"
echo "══════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Open Databricks workspace → /Shared/FLP/"
echo "  2. Run 00_setup_environment (creates catalog + tables)"
echo "  3. Run 01_preprocess → 02-05 train → 06_evaluate"
echo "  4. Or trigger the full workflow: databricks jobs run-now --job-id $JOB_ID"
echo ""
echo "To create serving endpoints (after first training):"
echo "  databricks serving-endpoints create --json-file databricks/serving/endpoint_config.json"
