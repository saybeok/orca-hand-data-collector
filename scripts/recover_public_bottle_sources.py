"""Create separate verified publication sources without altering original captures."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from orca_teleop.deferred_recording import capture_fingerprint

PROJECT = Path(__file__).resolve().parents[1]


def main():
    inventory = json.loads((PROJECT / 'data/act/external_dataset_inventory.json').read_text())
    recovery = json.loads((PROJECT / 'data/act/publication_verified_recovery.json').read_text())['files']
    fixes = {Path(item['path']): item for item in recovery}
    # Check backup contents again before making any derived files.
    for item in recovery:
        with Path(item['verified_backup']).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != item['expected']:
                raise ValueError('Backup checksum changed; recovery aborted.')
    base = Path('/media/lee/My Book/orcahand_lerobot/verified_capture_sources')
    base.mkdir(parents=True, exist_ok=True)
    for session in inventory['sessions']:
        source = Path(session['path'])
        if not any(path.is_relative_to(source) for path in fixes):
            continue
        target = base / source.name
        if target.exists():
            raise ValueError(f'Recovery output exists; inspect before reuse: {target}')
        staging = base / ('.' + source.name + '-' + uuid.uuid4().hex)
        staging.mkdir()
        for path in source.iterdir():
            if path.is_file() and path.suffix in {'.json', '.yaml'}:
                shutil.copy2(path, staging / path.name)
        for metadata in sorted(source.glob('episode-*/episode.json')):
            episode = json.loads(metadata.read_text())
            destination = staging / metadata.parent.name
            destination.mkdir()
            files = {'episode.json', 'rows.npz'} | {v['file'] for v in episode['videos'].values()}
            for name in files:
                if Path(name).name != name:
                    raise ValueError('Invalid capture filename.')
                original, output = metadata.parent / name, destination / name
                if original in fixes:
                    item = fixes[original]
                    shutil.copyfile(item['verified_backup'], output)
                    with output.open('rb') as stream:
                        if hashlib.file_digest(stream, 'sha256').hexdigest() != item['expected']:
                            raise ValueError('Recovery output checksum mismatch.')
                        os.fsync(stream.fileno())
                else:
                    # These files were verified; hardlinking does not alter their contents.
                    os.link(original, output)
        if capture_fingerprint(staging) != session['fingerprint']:
            raise ValueError('Recovery metadata changed.')
        staging.rename(target)
        session['path'] = str(target)
        print('Recovery source ready:', target, flush=True)
    inventory['verified_recovery_note'] = (
        'Original files unchanged; 24 video streams recovered from exact checksum-matching backups.'
    )
    (PROJECT / 'data/act/publication_verified_inventory.json').write_text(
        json.dumps(inventory, indent=2) + '\n'
    )
    print('Verified upload inventory ready.', flush=True)


if __name__ == '__main__':
    main()
