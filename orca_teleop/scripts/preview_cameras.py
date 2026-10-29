"""Live preview of OpenCV cameras before recording a dataset.

Use the same ``--camera NAME:INDEX[:WIDTHxHEIGHT][@FPS]`` specs as
``record_dataset.py``. Open a window per camera, physically aim them, then
press ``q`` / Esc to quit.

Examples::

    # Discover indices
    uv run python scripts/preview_cameras.py --list-cameras

    # Preview the iPhone Continuity Camera at index 1
    uv run python scripts/preview_cameras.py --camera iphone:1

    # Multiple cameras
    uv run python scripts/preview_cameras.py --camera front:0 --camera wrist:1

    # Linux USB ports: preserve physical identity when video numbers change
    uv run python scripts/preview_cameras.py --camera front:usb=1-1.1 --camera wrist:usb=1-1.3
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import subprocess
import sys
import time

import numpy as np

from orca_teleop.cameras import (
    OpenCVCamera,
    OpenCVCameraConfig,
    _capture_source,
    _linux_video_devices,
    list_available_cameras,
    parse_camera_spec,
    print_available_cameras,
)

logger = logging.getLogger(__name__)


def _print_camera_controls(name, source):
    """Read camera controls; never set focus, exposure, or other camera state."""
    print(f'\n=== {name}: {source} camera controls ===', flush=True)
    if not isinstance(source, str) or not source.startswith('/dev/video'):
        print('V4L2 control inspection is available for Linux video devices only.', flush=True)
        return
    command = shutil.which('v4l2-ctl')
    if not command:
        print('v4l2-ctl not installed; install v4l-utils to inspect focus support.', flush=True)
        return
    try:
        result = subprocess.run([command, '--device', source, '--list-ctrls'],
                                capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f'Could not inspect controls: {exc}', flush=True)
        return
    print(result.stdout.strip() or result.stderr.strip() or 'No controls reported.', flush=True)


def _open_previews(configs, tolerate_errors=False):
    cameras, errors = {}, {}
    for config in configs:
        camera = OpenCVCamera(config)
        try:
            camera.open()
        except BaseException as exc:
            camera.close()
            if not tolerate_errors or not isinstance(exc, Exception):
                for previous in cameras.values():
                    previous.close()
                raise
            errors[config.name] = str(exc)
            logger.warning('Skipping %s: %s', config.name, exc)
            continue
        cameras[config.name] = camera
    return cameras, errors


def _preview_grid(names, frames, labels, errors, tile_width=400, tile_height=300):
    """BGR contact sheet with persistent input labels and failure tiles."""
    import cv2

    columns = min(3, max(1, math.ceil(math.sqrt(len(names)))))
    rows = max(1, math.ceil(len(names)/columns))
    canvas = np.zeros((rows*tile_height, columns*tile_width, 3), dtype=np.uint8)
    for i, name in enumerate(names):
        y, x = (i//columns)*tile_height, (i % columns)*tile_width
        tile = canvas[y:y+tile_height, x:x+tile_width]
        rgb = frames.get(name)
        if rgb is not None:
            if rgb.dtype != np.uint8:
                rgb = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ratio = min(tile_width/bgr.shape[1], (tile_height-55)/bgr.shape[0])
            width, height = max(1, int(bgr.shape[1]*ratio)), max(1, int(bgr.shape[0]*ratio))
            resized = cv2.resize(bgr, (width, height))
            left, top = (tile_width-width)//2, 55+(tile_height-55-height)//2
            tile[top:top+height, left:left+width] = resized
        cv2.putText(tile, labels.get(name, name)[:52], (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, name, (8, 43), cv2.FONT_HERSHEY_SIMPLEX, .48,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if name in errors:
            cv2.putText(tile, 'NO LIVE IMAGE - see terminal', (8, tile_height//2),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 255), 1, cv2.LINE_AA)
    return canvas


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--list-cameras" in argv:
        print_available_cameras()
        return 0

    parser = argparse.ArgumentParser(
        description="Live-preview OpenCV cameras so you can aim them before recording."
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "Camera spec NAME[:INDEX|usb=USB_PATH][:WIDTHxHEIGHT][@FPS], repeatable. "
            "Same format as record_dataset.py."
        ),
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Probe available camera indices and exit.",
    )
    parser.add_argument('--all', dest='all_cameras', action='store_true',
                        help='Preview every readable video input in one labelled grid; failed inputs do not stop others.')
    parser.add_argument('--grid', action='store_true',
                        help='Display selected cameras together in one labelled window.')
    parser.add_argument('--arrange', action='store_true',
                        help='Arrange separate 640x480 preview windows in two columns.')
    parser.add_argument('--inspect-controls', action='store_true',
                        help='Print V4L2 focus/exposure controls for each selected camera without changing them.')
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Preview refresh rate (default: 30).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.all_cameras and args.camera:
        parser.error('use --all or --camera, not both')
    if not args.camera and not args.all_cameras:
        parser.error("pass --all, at least one --camera SPEC, or --list-cameras")

    if args.all_cameras:
        configs = [OpenCVCameraConfig(name=f'cam{index}', index=index, width=640, height=480, fps=15)
                   for index, _width, _height in list_available_cameras()]
        if not configs:
            logger.error('No readable video inputs found in this environment.')
            return 1
    else:
        configs = [parse_camera_spec(spec) for spec in args.camera]
    if len({c.name for c in configs}) != len(configs):
        parser.error('camera names must be unique')
    names = [c.name for c in configs]
    devices = _linux_video_devices()
    labels = {}
    for config in configs:
        try:
            source = _capture_source(config.index)
        except ValueError as exc:
            logger.error('%s: %s', config.name, exc)
            return 1
        index = int(source.removeprefix('/dev/video')) if isinstance(source, str) else source
        info = devices.get(index, {})
        usb = f"USB {info['usb_path']}" if info.get('usb_path') else ''
        label = info.get('interface', info.get('name', ''))
        labels[config.name] = f'{source} {usb} {label}'.strip()
        print(f'{config.name}: {labels[config.name]}', flush=True)
        if args.inspect_controls:
            _print_camera_controls(config.name, source)

    import cv2

    period = 1.0 / max(args.fps, 1e-3)
    cameras = {}
    try:
        cameras, open_errors = _open_previews(configs, tolerate_errors=args.all_cameras)
        if not cameras:
            logger.error('No camera inputs could be opened; close earlier previews and retry.')
            return 1
        grid = args.grid or args.all_cameras
        logger.info('Previewing %d live input(s); %d unavailable.', len(cameras), len(open_errors))
        print("Identify each view by cam<number>. Press 'q' or Esc to quit.")
        if args.arrange and not grid:
            for i, name in enumerate(cameras):
                title = f'preview: {name}'
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(title, 640, 480)
                cv2.moveWindow(title, (i % 2)*660, (i//2)*530)

        while True:
            t0 = time.monotonic()
            frames, errors = {}, dict(open_errors)
            for name, camera in cameras.items():
                try:
                    frames[name] = camera.read()
                except RuntimeError as exc:
                    if not args.all_cameras:
                        raise
                    errors[name] = str(exc)
            if grid:
                cv2.imshow('All camera inputs - q to close', _preview_grid(names, frames, labels, errors))
            else:
                for name, rgb in frames.items():
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"preview: {name}", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print()
    finally:
        for camera in cameras.values():
            camera.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
