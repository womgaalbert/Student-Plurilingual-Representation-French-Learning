"""
generate_roadmap_chart.py
MLOps & LLMOps Roadmap — French Learning Perceptions Project
Usage : python scripts/generate_roadmap_chart.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path("reports/roadmap_mlops_llmops.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "done"      : "#1B5E20",
    "done_light": "#C8E6C9",
    "wip"       : "#E65100",
    "wip_light" : "#FFE0B2",
    "todo_ml"   : "#0D47A1",
    "todo_ml_l" : "#BBDEFB",
    "todo_ll"   : "#4A148C",
    "todo_ll_l" : "#E1BEE7",
    "bg"        : "#F4F6F8",
    "header"    : "#1A1A2E",
    "subtext"   : "#546E7A",
    "text"      : "#212121",
    "line"      : "#B0BEC5",
    "arrow"     : "#78909C",
    "white"     : "#FFFFFF",
}

# ── Cards data ────────────────────────────────────────────────────────────────
mlops = [
    {
        "badge" : "LEVEL 0",  "tag": "Phase 1 — Termine",
        "title" : "Scripts Manuels",
        "status": "done",
        "bullet": "[OK]",
        "items" : [
            "preprocess.py",
            "train.py  (H1 -> H4)",
            "evaluate.py",
            "Rapports JSON / Word",
            "Logging Python",
        ],
    },
    {
        "badge" : "LEVEL 1",  "tag": "Phase 2 — Termine",
        "title" : "Pipeline Automatise",
        "status": "done",
        "bullet": "[OK]",
        "items" : [
            "pipeline.py  (10 etapes)",
            "MLflow Tracking",
            "GridSearchCV Tuning",
            "RandomOverSampler",
            "CamemBERT + UMAP",
        ],
    },
    {
        "badge" : "LEVEL 2",  "tag": "Phase 3 — A faire",
        "title" : "CI/CD & Deploiement",
        "status": "todo_ml",
        "bullet": "[ ]",
        "items" : [
            "GitHub Actions (train on push)",
            "MLflow Model Registry",
            "FastAPI /predict endpoints",
            "Docker + docker-compose",
            "Model Staging -> Production",
        ],
    },
    {
        "badge" : "LEVEL 3",  "tag": "Phase 4 — A faire",
        "title" : "Monitoring & Retraining",
        "status": "todo_ml",
        "bullet": "[ ]",
        "items" : [
            "Evidently AI  (data drift)",
            "Alertes drift automatiques",
            "Auto-retraining pipeline",
            "Health endpoint API",
            "Grafana Dashboard",
        ],
    },
]

llmops = [
    {
        "badge" : "PHASE A",  "tag": "En cours",
        "title" : "Embeddings & NLP",
        "status": "wip",
        "bullet": ">>>",
        "items" : [
            "CamemBERT features H3/H4",
            "PCA  768 -> 30 dims",
            "Semantic clustering UMAP",
            "Cosine similarity stereotypes",
            "A Priori classification",
        ],
    },
    {
        "badge" : "PHASE B",  "tag": "Phase 5 — A faire",
        "title" : "Prompt Engineering",
        "status": "todo_ll",
        "bullet": "[ ]",
        "items" : [
            "Few-shot classification",
            "Zero-shot hypotheses",
            "Chain-of-thought analyse",
            "Prompt templates FLES",
            "Evaluation qualitative LLM",
        ],
    },
    {
        "badge" : "PHASE C",  "tag": "Phase 6 — A faire",
        "title" : "RAG & Knowledge Base",
        "status": "todo_ll",
        "bullet": "[ ]",
        "items" : [
            "Vector store (FAISS/Chroma)",
            "Base documentaire FLES",
            "RAG pipeline pedagogique",
            "Reranking resultats",
            "Evaluation RAG (RAGAS)",
        ],
    },
    {
        "badge" : "PHASE D",  "tag": "Phase 7 — A faire",
        "title" : "LLM Agent & Rapport Auto",
        "status": "todo_ll",
        "bullet": "[ ]",
        "items" : [
            "Agent LLM analyse resultats",
            "Generation rapports auto",
            "Recommandations peda. LLM",
            "Fine-tuning CamemBERT",
            "Multilingual LLM (Africain)",
        ],
    },
]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 15), facecolor=C["bg"])

# ── Title block ───────────────────────────────────────────────────────────────
fig.text(0.5, 0.975,
         "MLOps  &  LLMOps  —  Roadmap  Projet  FLP",
         ha="center", va="top", fontsize=23, fontweight="bold",
         color=C["header"])
fig.text(0.5, 0.950,
         "French Learning Perceptions in Plurilingual Cameroon  |  "
         "Universite Marie & Louis Pasteur de Besancon",
         ha="center", va="top", fontsize=11, color=C["subtext"])

# Horizontal rule
fig.add_artist(plt.Line2D([0.03, 0.97], [0.933, 0.933],
               transform=fig.transFigure,
               color=C["line"], linewidth=1.8))

# ── Legend ────────────────────────────────────────────────────────────────────
legend_specs = [
    (C["done"],    C["done_light"],  "[OK]  Termine"),
    (C["wip"],     C["wip_light"],   "[>>>] En cours"),
    (C["todo_ml"], C["todo_ml_l"],   "[ ]   A faire  (MLOps)"),
    (C["todo_ll"], C["todo_ll_l"],   "[ ]   A faire  (LLMOps)"),
]
lx = 0.10
for edge, face, label in legend_specs:
    rect = FancyBboxPatch((lx, 0.908), 0.024, 0.018,
                          boxstyle="round,pad=0.003",
                          facecolor=face, edgecolor=edge,
                          linewidth=2,
                          transform=fig.transFigure, clip_on=False)
    fig.add_artist(rect)
    fig.text(lx + 0.028, 0.917, label,
             fontsize=9.5, color=C["text"], va="center")
    lx += 0.21

# ── Section headers ───────────────────────────────────────────────────────────
fig.text(0.255, 0.893,
         "━━━━━━━━━━━━   MLOps   ━━━━━━━━━━━━",
         ha="center", fontsize=13, fontweight="bold", color=C["done"])
fig.text(0.745, 0.893,
         "━━━━━━━━━━━━   LLMOps   ━━━━━━━━━━━━",
         ha="center", fontsize=13, fontweight="bold", color=C["todo_ll"])

# Vertical separator
fig.add_artist(plt.Line2D([0.505, 0.505], [0.085, 0.895],
               transform=fig.transFigure,
               color=C["line"], linewidth=2, linestyle="--"))

# ── Card drawing function ──────────────────────────────────────────────────────
def draw_card(left, bottom, width, height, data):
    s   = data["status"]
    col = C[s]
    lcl = C[s + "_light"] if s + "_light" in C else C["done_light"]

    ax = fig.add_axes([left, bottom, width, height])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off"); ax.set_facecolor(C["bg"])

    # Card body
    ax.add_patch(FancyBboxPatch(
        (0.03, 0.03), 0.94, 0.94,
        boxstyle="round,pad=0.025",
        facecolor=lcl, edgecolor=col, linewidth=2.5, zorder=1))

    # Header bar
    ax.add_patch(FancyBboxPatch(
        (0.03, 0.78), 0.94, 0.19,
        boxstyle="round,pad=0.01",
        facecolor=col, edgecolor=col, linewidth=0, zorder=2))

    # Badge (left of header)
    ax.text(0.07, 0.875, data["badge"],
            ha="left", va="center", fontsize=9.5,
            fontweight="bold", color=C["white"], zorder=3)

    # Tag (right of header)
    ax.text(0.93, 0.875, data["tag"],
            ha="right", va="center", fontsize=7.5,
            color=C["white"], alpha=0.90, zorder=3)

    # Title
    ax.text(0.50, 0.665,
            data["bullet"] + "  " + data["title"],
            ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=col, zorder=3)

    # Divider
    ax.add_artist(plt.Line2D([0.07, 0.93], [0.605, 0.605],
                  color=col, linewidth=0.8, alpha=0.4, zorder=3))

    # Items
    y = 0.545
    for item in data["items"]:
        ax.text(0.09, y, "->", ha="left", va="center",
                fontsize=8, color=col, alpha=0.75, zorder=3)
        ax.text(0.17, y, item, ha="left", va="center",
                fontsize=8.5, color=C["text"], zorder=3)
        y -= 0.105


# ── Layout geometry ───────────────────────────────────────────────────────────
CW  = 0.215   # card width
CH  = 0.355   # card height
GAP = 0.025   # gap between cards
TOP = 0.875   # top of first row

# MLOps  (left half) — 2 cols × 2 rows
mx = [0.030, 0.260]   # left edges
# LLMOps (right half) — 2 cols × 2 rows
lx2 = [0.520, 0.755]

rows_y = [TOP - CH, TOP - 2*CH - GAP]   # bottom edges of rows

for col_i, cards in enumerate([[mlops[0], mlops[2]],
                                [mlops[1], mlops[3]]]):
    for row_i, card in enumerate(cards):
        draw_card(mx[col_i], rows_y[row_i], CW, CH, card)

for col_i, cards in enumerate([[llmops[0], llmops[2]],
                                [llmops[1], llmops[3]]]):
    for row_i, card in enumerate(cards):
        draw_card(lx2[col_i], rows_y[row_i], CW, CH, card)

# ── Timeline bar ─────────────────────────────────────────────────────────────
bar = fig.add_axes([0.03, 0.055, 0.94, 0.09])
bar.set_xlim(0, 8); bar.set_ylim(0, 1)
bar.axis("off"); bar.set_facecolor(C["bg"])

bar.text(4, 0.96, "Chronologie d'execution du projet",
         ha="center", va="top", fontsize=10.5,
         fontweight="bold", color=C["header"])

timeline = [
    ("L0\nScripts",       C["done"],    C["done_light"]),
    ("L1\nPipeline",      C["done"],    C["done_light"]),
    ("LLMOps-A\nCamemBERT", C["wip"],  C["wip_light"]),
    ("L2\nCI/CD",         C["todo_ml"], C["todo_ml_l"]),
    ("L3\nMonitoring",    C["todo_ml"], C["todo_ml_l"]),
    ("LLMOps-B\nPrompts", C["todo_ll"], C["todo_ll_l"]),
    ("LLMOps-C\nRAG",     C["todo_ll"], C["todo_ll_l"]),
    ("LLMOps-D\nAgent",   C["todo_ll"], C["todo_ll_l"]),
]

for i, (label, edge, face) in enumerate(timeline):
    x = i + 0.08
    bar.add_patch(FancyBboxPatch(
        (x, 0.08), 0.78, 0.58,
        boxstyle="round,pad=0.02",
        facecolor=face, edgecolor=edge, linewidth=2))
    bar.text(x + 0.39, 0.37, label,
             ha="center", va="center",
             fontsize=7.8, fontweight="bold", color=edge)
    if i < len(timeline) - 1:
        bar.annotate("", xy=(x + 0.88, 0.37), xytext=(x + 0.78, 0.37),
                     arrowprops=dict(arrowstyle="->",
                                     color=C["arrow"], lw=1.8))

# "You are here" marker
bar.annotate("  Vous etes ici",
             xy=(2.47, 0.66), xytext=(2.47, 0.94),
             ha="center", fontsize=8.5,
             color=C["wip"], fontweight="bold",
             arrowprops=dict(arrowstyle="-|>",
                             color=C["wip"], lw=2.2))

# ── Footer ────────────────────────────────────────────────────────────────────
fig.add_artist(plt.Line2D([0.03, 0.97], [0.048, 0.048],
               transform=fig.transFigure,
               color=C["line"], linewidth=1.2))
fig.text(
    0.5, 0.026,
    "Projet FLP  |  Chercheuse : Chancelline Armelle Nongni Kendjio  "
    "(Doctorante FLES, Univ. Marie & Louis Pasteur, Besancon)  |  "
    "Support ML/AI : Albert Womga  |  Juin 2026",
    ha="center", fontsize=8.5, color=C["subtext"])

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig(str(OUT), dpi=180, bbox_inches="tight",
            facecolor=C["bg"], edgecolor="none")
plt.close()
print(f"Roadmap sauvegardee : {OUT}")
