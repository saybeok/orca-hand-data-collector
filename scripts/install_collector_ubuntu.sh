#!/usr/bin/env bash
# Run on the dedicated Ubuntu PC, after copying/extracting the project.
set -euo pipefail
task_root="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$task_root"

if ! command -v uv >/dev/null; then
  echo "uv를 먼저 설치하고 터미널을 다시 여세요: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
sudo apt-get update
sudo apt-get install -y python3-tk python3-numpy python3-opencv python3-yaml python3-pil.imagetk fonts-noto-cjk adb v4l-utils ffmpeg
sudo usermod -aG dialout,plugdev "$(id -un)"
sudo install -m 0644 configs/udev/70-orca-quest.rules /etc/udev/rules.d/70-orca-quest.rules
sudo udevadm control --reload-rules

# A virtualenv is machine/path-specific; do not copy .venv from the old PC.
uv sync --project orca_teleop --python 3.11 --extra learning
uv pip install --python orca_teleop/.venv/bin/python 'mujoco==3.6.0'
/usr/bin/python3 - <<'PY'
import tkinter
from pathlib import Path
root = Path.cwd()
target = Path.home() / '.local/share/applications/orca-collector.desktop'
target.parent.mkdir(parents=True, exist_ok=True)
executable = str(root / 'scripts/collector_ui.sh').replace('%', '%%').replace('\\', '\\\\').replace('"', '\\"').replace('`', '\\`').replace('$', '\\$')
target.write_text('[Desktop Entry]\nType=Application\nName=ORCA 시연 수집\n'
                  'Comment=Quest + AM 팔 + ORCA 손 LeRobot 수집\n'
                  f'Exec="{executable}"\nIcon=camera-video\nTerminal=false\nCategories=Utility;\n')
print('앱 메뉴에 ORCA 시연 수집을 등록했습니다.')
PY
echo "새 그룹 권한 적용을 위해 로그아웃 후 로그인하고 Quest USB를 다시 연결하세요."
echo "실행: $task_root/scripts/collector_ui.sh"
