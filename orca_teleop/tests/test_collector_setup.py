import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from orca_teleop.collector_app import Collector

spec = importlib.util.spec_from_file_location(
    'collector_setup_check', Path(__file__).parents[2] / 'scripts/check_collector_setup.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'new-user' / 'project with spaces'
    config_dir = root / 'configs/am-arm200-3dof'
    config_dir.mkdir(parents=True)
    config = {'home_pose_file': '../../data/arm_home/initial_pose.json',
              'port': '/dev/serial/by-id/same-robot',
              'joint_to_motor_map': {'shoulder_pan': 1, 'shoulder_lift': 2, 'elbow_flex': 3},
              'joint_signs': {'shoulder_pan': -1, 'shoulder_lift': 1, 'elbow_flex': 1}}
    (config_dir / 'config.yaml').write_text(yaml.safe_dump(config))
    reference = {'version': 1, 'motor_ids': [1, 2, 3],
                 'joint_names': ['shoulder_pan', 'shoulder_lift', 'elbow_flex'],
                 'joint_signs': [-1.0, 1.0, 1.0], 'port': config['port'],
                 'motor_rad': [-3.2, -1.0, -4.3]}
    (config_dir / 'initial_pose.json').write_text(json.dumps(reference, indent=2))
    return root


def test_new_pc_gets_same_arm_reference_without_changing_bytes(project):
    assert module.initialize_arm_home(project)
    assert (project / 'data/arm_home/initial_pose.json').read_bytes() == (
        project / 'configs/am-arm200-3dof/initial_pose.json').read_bytes()


def test_update_never_overwrites_an_already_taught_pose(project):
    target = project / 'data/arm_home/initial_pose.json'
    target.parent.mkdir(parents=True)
    target.write_text('existing locally taught pose')
    assert not module.initialize_arm_home(project)
    assert target.read_text() == 'existing locally taught pose'


def test_reference_for_another_arm_is_not_installed(project):
    path = project / 'configs/am-arm200-3dof/initial_pose.json'
    value = json.loads(path.read_text())
    value['joint_signs'][0] = 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='does not match'):
        module.initialize_arm_home(project)
    assert not (project / 'data/arm_home/initial_pose.json').exists()


def test_portable_defaults_follow_project_but_saved_settings_take_priority(project):
    defaults = {'task': 'move the bottle.', 'velocity': 30.0, 'acceleration': 90.0,
                'target': 50, 'fold_three_fingers': False}
    (project / 'configs/collector.defaults.json').write_text(json.dumps(defaults))
    collector = Collector(project)
    assert collector.settings['task'] == 'move the bottle.'
    assert collector.settings['velocity'] == 30.0
    assert collector.settings['data_root'] == str(project / 'data/lerobot')
    collector.save_settings(dict(collector.settings, task='custom task', velocity=10))
    assert Collector(project).settings['task'] == 'custom task'
    assert Collector(project).settings['velocity'] == 10
    assert Collector(project, demo=True).settings['task'] == 'move the eclipse case'


def test_machine_paths_cannot_be_imported_as_portable_defaults(project):
    (project / 'configs/collector.defaults.json').write_text(json.dumps({'data_root': '/old/pc'}))
    with pytest.raises(ValueError, match='저장 경로'):
        Collector(project)
