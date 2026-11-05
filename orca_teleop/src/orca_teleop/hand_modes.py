"""Optional hand command modes applied before the recording action mirror."""
from pathlib import Path

import numpy as np
import yaml
from orca_core import OrcaJointPositions


class FoldThreeFingersComposer:
    """Keep middle/ring/pinky folded while forwarding thumb, index and wrist.

    Wrapped before arm composition, so commands dispatched to the robot and
    mirrored into the dataset contain exactly the same folded finger targets.
    """

    def __init__(self, hand_config_path, composer=None):
        config = yaml.safe_load(Path(hand_config_path).read_text())
        self.composer = composer
        self.pose_deg = {}
        for finger in ('middle', 'ring', 'pinky'):
            for suffix, value in (('abd', config['neutral_position'][f'{finger}_abd']),
                                  ('mcp', 75.0), ('pip', 90.0)):
                joint = f'{finger}_{suffix}'
                limits = np.asarray(config['joint_roms'][joint], dtype=float)
                if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
                    raise ValueError(f'Invalid joint ROM for fixed finger: {joint}')
                self.pose_deg[joint] = float(np.clip(value, *limits))

    def reset(self):
        if self.composer is not None:
            self.composer.reset()

    def __call__(self, frame, hand_action):
        positions = dict(hand_action.data)
        positions.update(self.pose_deg)
        action = OrcaJointPositions(positions)
        return action if self.composer is None else self.composer(frame, action)
