"""Convert saved capture sessions into LeRobot datasets without opening hardware."""
import argparse
import logging
from pathlib import Path

from orca_teleop.collection_control import atomic_json
from orca_teleop.deferred_recording import capture_roots, convert_capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Capture session or parent folder.')
    parser.add_argument('--control-dir', type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s | %(message)s')
    state = {'phase': 'converting', 'saved': 0, 'frames': 0}
    def update(**fields):
        state.update(fields)
        if args.control_dir:
            args.control_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(args.control_dir / 'status.json', state)
    update()
    try:
        roots = capture_roots(args.source)
        if not roots:
            raise ValueError('선택한 폴더에 성공 저장된 수집본이 없습니다.')
        total_saved = 0
        for index, root in enumerate(roots):
            def progress(**fields):
                update(session=index + 1, sessions=len(roots), **fields)
            output = convert_capture(root, progress)
            import json
            total_saved += json.loads((output / 'conversion.json').read_text())['episodes']
            logging.info('LeRobot conversion complete: %s', output)
            update(dataset_root=str(output))
        update(phase='converted', saved=total_saved, error='')
    except Exception as exc:
        update(phase='error', error=f'변환 실패: {exc}. 수집 원본은 유지됩니다. 다시 변환할 수 있습니다.')
        logging.exception('Conversion failed; source captures retained.')
        raise


if __name__ == '__main__':
    main()
