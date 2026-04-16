# /project:train-h{1-4} | train-all

```powershell
# Une hypothèse (Level 0)
python src/train.py --hypothesis H1 --config params.yaml

# Pipeline complet (Level 1)
python src/pipeline.py --config params.yaml
```
Chaque run loggé dans MLflow automatiquement.
