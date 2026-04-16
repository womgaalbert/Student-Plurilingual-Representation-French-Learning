# Commandes Claude Code — French-Learning-Perceptions ML

## /project:setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## /project:preprocess
```bash
make preprocess
# ou directement :
python src/preprocessing/preprocess.py --input data/raw/data_FLP.csv --output data/processed
```

## /project:train-h1
**Charge skill** : `.claude/skills/h1-mobilisation/SKILL.md`
```bash
make train-h1
# Level 1 avec MLflow :
python pipelines/pipeline.py --hypothesis H1
```

## /project:train-h2
**Charge skill** : `.claude/skills/h2-representations/SKILL.md`
```bash
make train-h2
python pipelines/pipeline.py --hypothesis H2
```

## /project:train-h3
**Charge skill** : `.claude/skills/h3-exposition/SKILL.md`
```bash
make train-h3
python pipelines/pipeline.py --hypothesis H3
```

## /project:train-h4
**Charge skill** : `.claude/skills/h4-integration/SKILL.md`
```bash
make train-h4
python pipelines/pipeline.py --hypothesis H4
```

## /project:train-all
```bash
make train-all
# Level 1 complet (preprocess + train + evaluate + MLflow) :
python pipelines/pipeline.py --hypothesis ALL
```

## /project:evaluate
```bash
make evaluate
```

## /project:test
```bash
make test
```

## /project:mlflow
```bash
make mlflow-ui
# Ouvrir http://localhost:5000
```

## /project:status
Affiche l'état du projet :
```bash
python -c "
from src.utils.mlflow_utils import load_params, get_best_run
for h in ['H1','H2','H3','H4']:
    r = get_best_run(h)
    print(f'{h} : {r.get(\"metrics\", \"no runs yet\")}')
"
```
