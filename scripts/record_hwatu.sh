#!/usr/bin/env bash
# Real Quest + AM-ARM200/ORCA + three cameras -> local LeRobot dataset.
# Default: one manually-ended episode; append --num-episodes N for longer runs.
# --check-only opens cameras and checks adapters, without connecting to motors.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../env.sh"
cd "$ORCA_HAND_ROOT"
python_bin="$ORCA_HAND_ROOT/orca_teleop/.venv/bin/python"

if [[ " ${*} " == *" --help "* ]]; then
  exec "$python_bin" orca_teleop/scripts/record_dataset.py --help
fi

check_only=false
extra_args=()
for arg in "$@"; do
  if [[ "$arg" == --check-only ]]; then
    check_only=true
  else
    extra_args+=("$arg")
  fi
done

# Fail before enabling motor torque if a camera or adapter is missing.
"$python_bin" - <<'PY'
import os
from pathlib import Path
import yaml
from orca_teleop.cameras import CameraManager, parse_camera_spec

root = Path(os.environ['ORCA_HAND_ROOT'])
ports = []
for label, path in (
    ('hand', Path(os.environ['ORCA_MODEL_PATH'])),
    ('arm', root / 'configs/am-arm200-3dof/config.yaml'),
):
    port = Path(yaml.safe_load(path.read_text())['port'])
    if not port.exists():
        raise SystemExit(f'{label} USB adapter missing: {port}')
    ports.append(port.resolve())
    print(f'{label}: {port}', flush=True)
if ports[0] == ports[1]:
    raise SystemExit('Hand and arm must use different USB adapters.')

cameras = CameraManager([parse_camera_spec(os.environ[key]) for key in (
    'ORCA_CAMERA_BOTTOM', 'ORCA_CAMERA_WRIST', 'ORCA_CAMERA_REALSENSE')])
try:
    print('cameras:', cameras.open(), flush=True)
    cameras.ensure_live(samples=5, interval_s=0.1)
finally:
    cameras.close()
print('Camera/USB checks passed. No motors were connected.', flush=True)
PY

if "$check_only"; then
  exit 0
fi

if ! adb reverse tcp:8765 tcp:8765; then
  echo "Quest ADB 연결 실패. adb devices -l로 권한/USB 디버깅 허용 상태를 확인하세요." >&2
  echo "insufficient permissions: README.md의 Quest USB 권한 설정을 적용하세요." >&2
  exit 1
fi
session_id="hwatu_$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)_$$"
dataset_root="$ORCA_HAND_ROOT/data/lerobot/$session_id"
mkdir -p "$ORCA_HAND_ROOT/data/lerobot"
echo "Quest Browser: http://localhost:8765 → Start hand tracking"
echo "Left pinch: arm clutch. SPACE: save/next. q/Esc: save current episode and finish."
echo "From episode 2, the robot returns home automatically before recording."
echo "Dataset: $dataset_root"

exec "$python_bin" orca_teleop/scripts/record_dataset.py \
  --backend am200_hw --local --source metaquest --hand right \
  --hand-config "$ORCA_MODEL_PATH" --urdf-path "$ORCA_URDF_PATH" \
  --retargeter adaptive_analytical \
  --retarget-config "$ORCA_HAND_ROOT/configs/adaptive_analytical_orca_v1_pinch_calibrated.yaml" \
  --clutch --arm-scale 0.35 \
  --arm-max-velocity-deg-s 15 --arm-max-acceleration-deg-s2 45 \
  --fps 30 --episode-end space --num-episodes 1 \
  --task "move the hwatu card." \
  --repo-id "local/$session_id" --root "$dataset_root" \
  --camera "$ORCA_CAMERA_BOTTOM" \
  --camera "$ORCA_CAMERA_WRIST" \
  --camera "$ORCA_CAMERA_REALSENSE" \
  "${extra_args[@]}"
