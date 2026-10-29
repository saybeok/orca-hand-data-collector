#!/usr/bin/env bash
# Bring-up wrapper around the orca_core scripts for the v1 Feetech right hand.
#
#   ./hand.sh detect              show the serial port and what orca_core detects
#   ./hand.sh configure           burn IDs into a daisy-chained set of motors
#   ./hand.sh setid N             burn ID N into the single motor on the bus
#   ./hand.sh tension             hold the hand taut for manual tendon setup
#   ./hand.sh tension --joints thumb_dip --no-move-motors
#                                 hold only thumb IP at its current position
#   ./hand.sh calibrate           learn the motor limits (writes calibration.yaml)
#   ./hand.sh neutral             move to the neutral pose
#   ./hand.sh test-thumb          slowly move only the thumb IP motor and back
#   ./hand.sh test-joint --joint thumb_pip --delta-deg 20 --speed-deg-s 5
#                                 slowly probe one joint without changing calibration
#   ./hand.sh contacts capture     record manually confirmed pad contacts (finger torque off)
#   ./hand.sh contacts fit         validate a separate contact-point retarget profile offline
#   ./hand.sh contacts fit-surface fit empirical skin clearances for continuous retargeting
#   ./hand.sh contacts teach       use recorded contact configurations as pinch targets
#   ./hand.sh recenter --motors N [N ...] [--apply]
#                                 move a motor's encoder seam off its travel
#
# Append --mock to tension, calibrate or neutral for a dry run with no hardware.
# `configure` is a one-time step for a freshly assembled hand; motors that
# already carry their IDs are skipped, so re-running it is safe.
set -euo pipefail

source "$(dirname "$(readlink -f "$0")")/../env.sh"

PY="$ORCA_HAND_ROOT/orca_teleop/.venv/bin/python"
cmd="${1:-}"
shift || true

case "$cmd" in
  detect)
    # This hand's CH340 variant (1a86:55d3) binds cdc-acm, so it shows up as
    # /dev/ttyACM0 rather than the /dev/ttyUSB0 of the older CH340 chips.
    shopt -s nullglob
    devices=(/dev/ttyUSB* /dev/ttyACM*)
    shopt -u nullglob
    if (( ${#devices[@]} )); then
      ls -l "${devices[@]}"
    else
      echo "no serial device found"
    fi
    lsusb | grep -iE "1a86|10c4" || echo "no Feetech USB adapter on the bus"
    exec "$PY" -c "
from orca_core.constants import MOTOR_BAUD_RATES
from orca_core.maintenance import detect_motor_type, resolve_port
from orca_core.maintenance.motor_chain import scan_motors

port = resolve_port('auto', None)
print(f'port        : {port}')
if port is None:
    raise SystemExit(1)

# Answers only at a factory default, so a fully configured chain reports None.
motor_type = detect_motor_type(port) or 'feetech'
print(f'motor_type  : {motor_type}')

for baud in MOTOR_BAUD_RATES[motor_type]:
    found = scan_motors(motor_type, port, baud, (1, 17))
    if found:
        print(f'motors @ {baud:,} baud:')
        for m in found:
            print(f\"  ID {m['id']:2d}  {m['model_name']}\")
"
    ;;
  configure)
    cd "$ORCA_HAND_ROOT/orca_core"
    exec "$PY" scripts/configure_motor_chain.py "$ORCA_MODEL_PATH" "$@"
    ;;
  setid)
    exec "$PY" "$ORCA_HAND_ROOT/scripts/set_motor_id.py" "$@"
    ;;
  recenter)
    exec "$PY" "$ORCA_HAND_ROOT/scripts/recenter_motor_offset.py" "$@"
    ;;
  test-thumb|test-joint)
    exec "$PY" "$ORCA_HAND_ROOT/scripts/check_thumb_ip.py" "$@"
    ;;
  contacts)
    exec "$PY" "$ORCA_HAND_ROOT/scripts/calibrate_hand_contacts.py" "$@"
    ;;
  tension|calibrate|neutral)
    cd "$ORCA_HAND_ROOT/orca_core"
    exec "$PY" "scripts/$cmd.py" "$ORCA_MODEL_PATH" "$@"
    ;;
  *)
    echo "usage: $(basename "$0") {detect|configure|setid|recenter|tension|calibrate|neutral|test-thumb|test-joint|contacts} [args]" >&2
    exit 2
    ;;
esac
