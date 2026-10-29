"""Program ONE motor at a time, with only that motor on the bus.

configure_motor_chain.py requires every already-configured motor to stay
daisy-chained, because it re-verifies the whole chain after each step. This
script drops that requirement: connect a single factory-default motor, give it
its target ID, unplug it, repeat. Label each motor as you go, then wire them in
descending ID order (board -> 17 -> 16 -> ... -> 1) once the hand is assembled.

Usage, with the hand config's motor layout as the source of truth:

    python scripts/set_motor_id.py 17          # program the connected motor to ID 17
    python scripts/set_motor_id.py 17 --force  # skip the model-name check
    python scripts/set_motor_id.py --plan      # print which ID expects which model
    python scripts/set_motor_id.py --scan      # show what is on the bus right now
    python scripts/set_motor_id.py 3 --reset   # send ID 3 back to factory defaults
"""

import argparse
import os
import sys
import time

from orca_core.maintenance import plan_motor_chain, resolve_port
from orca_core.maintenance.motor_chain import (
    ID_CHANGE_SETTLE_S,
    MotorChainError,
    _config_session,
    reset_motor_to_factory,
    scan_motors,
)
from orca_core.utils import read_yaml

GREEN, RED, YELLOW, BOLD, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[1m", "\033[0m"


def build_plan(port: str):
    config_path = os.environ["ORCA_MODEL_PATH"]
    config = read_yaml(config_path)
    motor_type = config.get("motor_type") or "feetech"
    return plan_motor_chain(config, port, motor_type)


def scan_all(plan) -> list[dict]:
    """Motors answering at either the target or the factory baud."""
    seen: dict[tuple[int, int], dict] = {}
    for baud in dict.fromkeys((plan.target_baud, plan.default_baud)):
        for m in scan_motors(plan.motor_type, plan.port, baud, (0, 253)):
            seen[(m["id"], m["baud_rate"])] = m
    return [seen[k] for k in sorted(seen)]


def cmd_plan(plan) -> int:
    print(f"{BOLD}chain order (board first){RESET}")
    for target_id in plan.all_target_ids:
        role = "wrist " if target_id == plan.wrist_id else "finger"
        print(f"  ID {target_id:2d}  {role}  {plan.model_for(target_id)}")
    print(f"\ntarget baud     : {plan.target_baud:,}")
    print(f"factory default : ID {plan.default_id} @ {plan.default_baud:,}")
    return 0


def cmd_scan(plan) -> int:
    motors = scan_all(plan)
    if not motors:
        print(f"{YELLOW}no motor answered on {plan.port}{RESET}")
        return 1
    for m in motors:
        print(f"  ID {m['id']:3d}  {m['model_name']:10s} @ {m['baud_rate']:,} baud")
    return 0


def cmd_reset(plan, target_id: int) -> int:
    match = [m for m in scan_all(plan) if m["id"] == target_id]
    if not match:
        print(f"{RED}no motor with ID {target_id} on the bus{RESET}")
        return 1
    reset_motor_to_factory(plan, target_id, match[0]["baud_rate"])
    print(f"{GREEN}ID {target_id} reset to ID {plan.default_id} @ {plan.default_baud:,}{RESET}")
    return 0


def cmd_program(plan, target_id: int, force: bool) -> int:
    if target_id not in plan.all_target_ids:
        print(f"{RED}ID {target_id} is not in this hand's layout {plan.all_target_ids}{RESET}")
        return 1

    motors = scan_all(plan)
    if not motors:
        print(f"{RED}no motor answered on {plan.port}. Check power and wiring.{RESET}")
        return 1
    if len(motors) > 1:
        listed = ", ".join(f"ID {m['id']} @ {m['baud_rate']:,}" for m in motors)
        print(f"{RED}{len(motors)} motors on the bus ({listed}).{RESET}")
        print("Programming one at a time needs exactly one motor connected.")
        return 1

    motor = motors[0]
    expected = plan.model_for(target_id)
    print(f"found: ID {motor['id']} {motor['model_name']} @ {motor['baud_rate']:,} baud")

    if expected not in motor["model_name"]:
        note = f"ID {target_id} expects {expected}, but a {motor['model_name']} is connected"
        if not force:
            print(f"{RED}{note}. Swap the motor, or pass --force.{RESET}")
            return 1
        print(f"{YELLOW}{note} — continuing because of --force.{RESET}")

    if motor["baud_rate"] != plan.target_baud:
        with _config_session(
            plan.motor_type, [motor["id"]], plan.port, motor["baud_rate"]
        ) as client:
            if not client.change_motor_baudrate(motor["id"], plan.target_baud):
                raise MotorChainError(f"failed to set baud rate on ID {motor['id']}")
        time.sleep(ID_CHANGE_SETTLE_S)
        print(f"  baud {motor['baud_rate']:,} -> {plan.target_baud:,}")

    if motor["id"] != target_id:
        with _config_session(
            plan.motor_type, [motor["id"]], plan.port, plan.target_baud
        ) as client:
            if not client.change_motor_id(motor["id"], target_id):
                raise MotorChainError(f"failed to set ID {target_id}")
        time.sleep(ID_CHANGE_SETTLE_S)
        print(f"  ID {motor['id']} -> {target_id}")

    after = [m for m in scan_all(plan) if m["id"] == target_id]
    if not after or after[0]["baud_rate"] != plan.target_baud:
        print(f"{RED}verification failed: ID {target_id} did not answer at "
              f"{plan.target_baud:,} baud{RESET}")
        return 1

    print(f"{GREEN}{BOLD}✓ ID {target_id} @ {plan.target_baud:,} baud "
          f"({after[0]['model_name']}){RESET}")
    print(f"{YELLOW}Label this motor '{target_id}' before disconnecting it.{RESET}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("target_id", type=int, nargs="?", help="ID to program into the motor")
    parser.add_argument("--plan", action="store_true", help="print the ID/model layout and exit")
    parser.add_argument("--scan", action="store_true", help="list motors on the bus and exit")
    parser.add_argument("--reset", action="store_true", help="revert TARGET_ID to factory defaults")
    parser.add_argument("--force", action="store_true", help="program despite a model mismatch")
    parser.add_argument("--port", default=None, help="serial port (default: auto-detect)")
    args = parser.parse_args()

    config = read_yaml(os.environ["ORCA_MODEL_PATH"])

    # --plan only reports the config's layout, so it must work with nothing plugged in.
    if args.plan:
        return cmd_plan(build_plan("unused"))

    port = args.port or resolve_port("auto", config.get("motor_type"))
    if port is None:
        print(f"{RED}no serial port found. Is the hand plugged in?{RESET}")
        return 1
    plan = build_plan(port)
    print(f"port: {port}  motors: {plan.motor_type}\n")

    if args.scan:
        return cmd_scan(plan)
    if args.target_id is None:
        parser.error("TARGET_ID is required unless --plan or --scan is given")
    if args.reset:
        return cmd_reset(plan, args.target_id)
    return cmd_program(plan, args.target_id, args.force)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{RED}interrupted{RESET}")
        sys.exit(1)
    except MotorChainError as exc:
        print(f"{RED}{exc}{RESET}")
        sys.exit(1)
