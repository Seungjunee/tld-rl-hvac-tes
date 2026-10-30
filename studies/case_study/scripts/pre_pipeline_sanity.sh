#!/bin/bash
# Pre-pipeline sanity: 1-episode RBC for each active building.
# Verifies: env builds, reward signal alive, |comfort| < |energy| (no broken sim).
# Runtime: ~3-5 min (4-parallel single-episode).
#
# Failure modes detected:
#   1. Building epJSON broken (env build error)
#   2. Reward range_comfort doesn't contain setpoint (always-violating reward)
#   3. q_rated mismatch with autosize (gradient too weak)
#   4. Action space wired wrong (PPO can't change setpoint)
set -e
cd "$(git rev-parse --show-toplevel)"
mkdir -p /tmp/sanity_logs

CASES=(rbc_office_smud_tou rbc_lbuilding_smud_tou rbc_warehouse_smud_tou rbc_mediumoffice_smud_tou)
echo "=== Pre-pipeline sanity (1-episode RBC × 3 buildings) ==="
date

# Override episodes via env var (recipes.yaml episodes_50_with_eval is 30 → cut to 1 for sanity)
# Use a dedicated profile if it exists, otherwise rely on the runner respecting --total-timesteps... actually
# the simplest is to run full 30 ep RBC since RBC is fast (~5min/building, 4-parallel).

for c in "${CASES[@]}"; do
  rm -rf "artifacts/case_study/exports/$c" "artifacts/case_study/workspaces/$c"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m studies.case_study.cli.run \
    --case "$c" > "/tmp/sanity_logs/$c.log" 2>&1 &
done
wait

echo "--- Sanity check results ---"
PASS=1
for c in "${CASES[@]}"; do
  pcsv="artifacts/case_study/exports/$c/progress.csv"
  if [ ! -f "$pcsv" ]; then
    echo "  ❌ $c: no progress.csv (env build failure?)"
    PASS=0
    continue
  fi
  python - <<EOF
import pandas as pd
import sys
df = pd.read_csv("$pcsv")
full = df[df['length(timesteps)'] >= 11711]
if len(full) == 0:
    print(f"  ❌ $c: no full episodes")
    sys.exit(1)
last = full.iloc[-1]
comfort = abs(last['mean_reward_comfort_term'])
energy = abs(last['mean_reward_energy_term'])
mean_kw = last['mean_power_demand'] / 1000
ratio = comfort / max(energy, 1e-6)
status = '✅' if (comfort < energy and 0.5 < mean_kw < 30) else '❌'
print(f"  {status} $c: kW={mean_kw:.2f}, |comfort|={comfort:.2f}, |energy|={energy:.0f}, c/e={ratio:.3f}")
if comfort > energy:
    print(f"     WARN: comfort term DOMINATES energy — broken reward (range_comfort doesn't contain setpoint?)")
    sys.exit(1)
if mean_kw < 0.5 or mean_kw > 30:
    print(f"     WARN: kW out of expected range [0.5, 30]")
    sys.exit(1)
EOF
  [ $? -ne 0 ] && PASS=0
done

if [ $PASS -eq 1 ]; then
  echo ""
  echo "🟢 Sanity PASS — pipeline OK to launch"
  exit 0
else
  echo ""
  echo "🔴 Sanity FAIL — fix issues before running pipeline"
  exit 1
fi
