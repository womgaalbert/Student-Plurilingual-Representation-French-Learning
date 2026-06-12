"""
prepare_cloud_deploy.py — Prepare le projet pour Streamlit Community Cloud
Copie les fichiers necessaires dans un dossier de deployement allege.

Usage : python scripts/prepare_cloud_deploy.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEPLOY = ROOT / "deploy_cloud"

# Nettoyer
if DEPLOY.exists():
    shutil.rmtree(DEPLOY)
DEPLOY.mkdir()

# Fichiers racine
for f in ["app.py", "params.yaml", "requirements-cloud.txt"]:
    src = ROOT / f
    if src.exists():
        shutil.copy2(src, DEPLOY / f)
        print(f"  COPY {f}")

# Renommer requirements-cloud.txt -> requirements.txt (Streamlit Cloud le cherche)
cloud_req = DEPLOY / "requirements-cloud.txt"
if cloud_req.exists():
    cloud_req.rename(DEPLOY / "requirements.txt")
    print("  RENAME requirements-cloud.txt -> requirements.txt")

# Modeles — seulement les derniers tuned (1 par hypothese/sous-modele)
MODEL_DIR = DEPLOY / "models"
_latest = lambda path, pat: sorted(
    Path(path).glob(pat), key=lambda p: p.stat().st_mtime, reverse=True
)

model_specs = [
    ("h1", "*.pkl", 1),                              # H1: XGBoost
    ("h2", "*A_motivation_tuned*.pkl", 1),            # H2: motivation pipeline
    ("h3", "*reg_tuned*.pkl", 1),                     # H3: regression
    ("h3", "*clf_tuned*.pkl", 1),                     # H3: classification
    ("h4", "*A_motivation_tuned*.pkl", 1),            # H4: motivation
    ("h4", "*B_engagement_tuned*.pkl", 1),            # H4: engagement
    ("h4", "*C_discipline_tuned*.pkl", 1),            # H4: disciplines
]

for hyp, pattern, count in model_specs:
    src_dir = ROOT / "models" / hyp
    dst_dir = MODEL_DIR / hyp
    dst_dir.mkdir(parents=True, exist_ok=True)
    models = _latest(src_dir, pattern)
    for m in models[:count]:
        shutil.copy2(m, dst_dir / m.name)
        size_kb = m.stat().st_size / 1024
        print(f"  MODEL {hyp}/{m.name} ({size_kb:.0f} KB)")

# Images descriptives (optionnel — leger)
reports_src = ROOT / "reports"
reports_dst = DEPLOY / "reports"
for sub in ["demographics", "h1/descriptive", "h2/descriptive", "h3/descriptive", "h4/descriptive"]:
    src_sub = reports_src / sub
    if not src_sub.exists():
        continue
    dst_sub = reports_dst / sub
    dst_sub.mkdir(parents=True, exist_ok=True)
    for img in sorted(src_sub.glob("*.png")):
        shutil.copy2(img, dst_sub / img.name)
    count = len(list(dst_sub.glob("*.png")))
    print(f"  IMAGES {sub}/ ({count} png)")

# Source code — minimum pour les imports
src_dst = DEPLOY / "src"
src_dst.mkdir()
for f in ["__init__.py"]:
    shutil.copy2(ROOT / "src" / f, src_dst / f)

utils_dst = src_dst / "utils"
utils_dst.mkdir()
for f in ["__init__.py", "config.py", "constants.py"]:
    shutil.copy2(ROOT / "src" / "utils" / f, utils_dst / f)
print("  SRC src/utils/{config,constants}.py")

# .streamlit config
streamlit_dst = DEPLOY / ".streamlit"
streamlit_dst.mkdir()
shutil.copy2(ROOT / ".streamlit" / "config.toml", streamlit_dst / "config.toml")
print("  CONFIG .streamlit/config.toml")

# Taille totale
total = sum(f.stat().st_size for f in DEPLOY.rglob("*") if f.is_file())
print(f"\nTotal deployement : {total / 1024 / 1024:.1f} MB")
print(f"Dossier : {DEPLOY}")
