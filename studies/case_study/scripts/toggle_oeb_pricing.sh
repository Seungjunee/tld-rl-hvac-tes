#!/bin/bash
# Toggle reward profiles between OEB 1x and OEB 2x pricing.
# Updates 5 reward profiles in recipes.yaml via single sed.
#
# Usage:
#   bash studies/case_study/scripts/toggle_oeb_pricing.sh 1x
#   bash studies/case_study/scripts/toggle_oeb_pricing.sh 2x
#
# After toggle, all transfer/RBC results from previous pricing are INVALIDATED.
# Re-run pipeline_3building to regenerate.
set -e
cd "$(git rev-parse --show-toplevel)"

TARGET="${1:-}"
RECIPES="studies/case_study/config/recipes.yaml"

case "$TARGET" in
  1x)
    FROM="*oeb_2x"
    TO="*oeb_1x"
    ;;
  2x)
    FROM="*oeb_1x"
    TO="*oeb_2x"
    ;;
  *)
    echo "Usage: $0 {1x|2x}"
    echo ""
    echo "Current state:"
    n2x=$(grep -cF "price_schedule: *oeb_2x" "$RECIPES" || true)
    n1x=$(grep -cF "price_schedule: *oeb_1x" "$RECIPES" || true)
    echo "  OEB 2x profiles: $n2x"
    echo "  OEB 1x profiles: $n1x"
    exit 1
    ;;
esac

# sed in-place; escape * (regex metachar)
FROM_ESC=$(printf '%s' "$FROM" | sed 's/\*/\\*/g')
TO_ESC=$(printf '%s' "$TO" | sed 's/\*/\\*/g')
sed -i.bak "s|price_schedule: ${FROM_ESC}|price_schedule: ${TO}|g" "$RECIPES"
rm -f "$RECIPES.bak"

n_from=$(grep -cF "price_schedule: $FROM" "$RECIPES" || true)
n_to=$(grep -cF "price_schedule: $TO" "$RECIPES" || true)
echo "Toggled to OEB $TARGET"
echo "  $TO profiles: $n_to (expected 5)"
echo "  $FROM profiles: $n_from (expected 0)"

if [ "$n_to" -ne 5 ] || [ "$n_from" -ne 0 ]; then
  echo "WARN: profile count mismatch. Inspect manually:"
  grep -n "price_schedule:" "$RECIPES"
  exit 1
fi

# Verify resolved schedule
echo ""
echo "Verifying resolved price_schedule:"
python -m studies.case_study.cli.inspect --case rbc_office_tou 2>&1 | \
  python -c "import json, sys, re; m=re.search(r'\"price_schedule\":\s*\[(.*?)\]', sys.stdin.read(), re.S); vals=[float(v) for v in m.group(1).split(',') if v.strip()] if m else []; peak=max(vals); off=min(vals); ratio=peak/off; print(f'  peak={peak}, off={off}, ratio={ratio:.2f}x')"
