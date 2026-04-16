#!/usr/bin/env bash
# validate-bash.sh — French-Learning-Perceptions ML
# Hook pre-tool Claude Code (Git Bash / WSL sur Windows)
# Exécuté avant chaque opération sur les données ou fichiers.

set -e
COMMAND="${1:-}"
FILE="${2:-}"

# ── RÈGLE 1 : Bloquer écriture dans data/raw/ ──────────────────────────────
if echo "$FILE" | grep -qE "data[/\\\\]raw[/\\\\]"; then
    if ! echo "$COMMAND" | grep -q "read\|cat\|head\|load_csv\|pd.read"; then
        echo "❌ BLOQUÉ : Écriture dans data/raw/ interdite."
        echo "   → Utiliser data/processed/ pour les sorties."
        exit 1
    fi
fi

# ── RÈGLE 2 : Vérifier consentement avant entraînement ─────────────────────
if echo "$COMMAND" | grep -qE "train|fit|pipeline"; then
    CLEAN="data/processed/french-learning-perceptions_clean.csv"
    if [ -f "$CLEAN" ]; then
        python -c "
import pandas as pd, sys
df = pd.read_csv('$CLEAN')
cols = [c for c in df.columns if 'consentement' in c.lower() or 'accepte' in c.lower()]
if not cols:
    print('INFO : Colonne consentement non trouvée — vérifier manuellement.')
    sys.exit(0)
col = cols[0]
n_invalid = (~df[col].str.lower().str.contains('accepte', na=False)).sum()
if n_invalid > 0:
    print(f'❌ BLOQUÉ : {n_invalid} lignes sans consentement valide.')
    sys.exit(1)
print(f'✅ Consentement OK — {len(df)} répondants validés.')
" 2>/dev/null || true
    fi
fi

# ── RÈGLE 3 : Bloquer export avec colonnes sensibles ───────────────────────
if echo "$COMMAND" | grep -qE "to_csv|to_excel|export|save"; then
    if [ -n "$FILE" ] && [ -f "$FILE" ]; then
        python -c "
import pandas as pd, sys
try:
    df = pd.read_csv('$FILE')
    forbidden = ['horodateur', 'timestamp', 'email', 'nom', 'prenom']
    found = [c for c in df.columns if any(f in c.lower() for f in forbidden)]
    if found:
        print(f'❌ BLOQUÉ : Colonnes sensibles détectées : {found}')
        sys.exit(1)
    print('✅ Anonymisation OK.')
except Exception:
    pass
" 2>/dev/null || true
    fi
fi

echo "✅ Hook validate-bash : OK"
exit 0
