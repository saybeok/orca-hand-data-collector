

import argparse
from dataclasses import replace

from orca_core import MockOrcaHand, OrcaHand
from orca_core.hand_config import OrcaHandConfig

from orca_core.utils.cli import add_hand_arguments, connect_hand, create_hand, shutdown_hand


def _selected_joint_config(config: OrcaHandConfig, joints: list[str]) -> OrcaHandConfig:
    """Restrict the client itself, including connect/disconnect, to these motors."""
    selected = list(dict.fromkeys(joints))
    if not selected:
        raise ValueError("Select at least one joint.")
    unknown = set(selected) - set(config.joint_ids)
    if unknown:
        raise ValueError(f"Unknown joint(s): {sorted(unknown)}")
    missing = [joint for joint in selected if joint not in config.joint_to_motor_map]
    if missing:
        raise ValueError(f"Joint(s) have no motor mapping: {missing}")
    mapping = {joint: config.joint_to_motor_map[joint] for joint in selected}
    motor_ids = set(mapping.values())
    return replace(
        config,
        joint_ids=selected,
        motor_ids=[mid for mid in config.motor_ids if mid in motor_ids],
        joint_to_motor_map=mapping,
        joint_inversion_dict={joint: config.joint_inversion_dict.get(joint, False) for joint in selected},
        joint_roms_dict={joint: config.joint_roms_dict[joint] for joint in selected},
        neutral_position={joint: config.neutral_position[joint] for joint in selected if joint in config.neutral_position},
        calibration_sequence=[],
        calibration_min_motor_span_rad={joint: span for joint, span in config.calibration_min_motor_span_rad.items()
                                       if joint in selected},
        joint_encoder_joints=None,
        use_joint_feedback=False,
    )


def _preload_current_position(hand) -> None:
    # Mode changes enable torque. Replace any stale goal with a fresh measured
    # position first, so selected-joint setup cannot jump to an old goal.
    positions = hand._read_motor_pos_for_offsets()
    validate_frame = getattr(hand.motor_client, "validate_position_frame", None)
    if validate_frame is not None:
        validate_frame(hand.config.motor_ids, positions)
    hand._set_motor_pos(positions)


def _print_progress(event: dict) -> None:
    """Render tension progress events on the terminal."""
    phase = event.get("phase") if event.get("event") == "phase" else None
    if phase == "winding":
        print("Winding tendons taut...")
    elif phase == "holding":
        print("Holding motors for manual tensioning — press Ctrl+C when done.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hold the hand under tension for manual tendon setup."
    )
    add_hand_arguments(parser)
    parser.add_argument(
        "--joints", nargs="+",
        help="Only connect and tension these joints (e.g. --joints thumb_dip).",
    )
    parser.add_argument(
        "--move-motors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the built-in preconditioning motion before holding tension (default: True).",
    )
    args = parser.parse_args()

    if args.joints:
        try:
            config = _selected_joint_config(
                OrcaHandConfig.from_config_path(args.config_path), args.joints,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.move_motors and any("wrist" in joint.lower() for joint in config.joint_ids):
            parser.error("Wrist winding is unsupported; use --no-move-motors to hold it only.")
        hand_cls = MockOrcaHand if args.mock else OrcaHand
        hand = hand_cls(config=config)
        print("Tension motor mapping: " + ", ".join(
            f"{joint}=ID {motor}" for joint, motor in config.joint_to_motor_map.items()
        ))
    else:
        hand = create_hand(args.config_path, use_mock=args.mock)
    try:
        connect_hand(hand)
        if args.joints:
            _preload_current_position(hand)
        hand.tension(move_motors=args.move_motors, progress_callback=_print_progress)
        return 0
    except KeyboardInterrupt:
        print("\nTension interrupted.")
        return 0
    finally:
        shutdown_hand(hand)

if __name__ == "__main__":
    raise SystemExit(main())
