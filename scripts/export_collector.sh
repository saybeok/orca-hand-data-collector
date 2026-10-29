#!/usr/bin/env bash
# Include locally modified controllers, models, mapping and contact calibration.
# Exclude machine-specific virtualenvs, git metadata and collected datasets.
set -euo pipefail
task_root="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
mkdir -p "$task_root/data/exports"
archive="$task_root/data/exports/orca_collector_$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$archive" -C "$task_root" \
  --exclude='.git' --exclude='.git.upstream' --exclude='.venv' --exclude='__pycache__' \
  --exclude='.pytest_cache' --exclude='.ruff_cache' --exclude='*.egg-info' \
  env.sh README.md scripts configs orca_core orca_teleop orca_am200 \
  orcahand_description data/contact_calibration
echo "$archive"
