"""Run the local bottle ACT checkpoint with its saved processors and joint frame.

Default: check cameras and infer once, without connecting to motors.
--offline checks a held-out dataset observation instead of opening cameras.
--execute drives the physical 20-DoF robot; no Quest connection is needed.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / 'data/act/bottle_20261108/capture/lerobot'
DEFAULT_POLICY = ROOT / 'outputs/train/act_bottle_20261108_154509/checkpoints/last/pretrained_model'
LOG = logging.getLogger('act_inference')


class ACTRuntime:
    def __init__(self, checkpoint: Path, dataset: Path, device: str):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.torch = torch
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA를 사용할 수 없습니다. PC 터미널에서 실행하거나 --device cpu를 지정하세요.')
        self.device = device
        training = json.loads((checkpoint / 'train_config.json').read_text())
        self.repo_id = training['dataset']['repo_id']
        self.meta = json.loads((dataset / 'meta/info.json').read_text())
        self.session = json.loads((dataset / 'collection_session.json').read_text())
        self.joints = self.meta['features']['action']['names']
        if self.meta['features']['observation.state']['names'] != self.joints or len(self.joints) != 20:
            raise ValueError('Dataset state/action joint order is not the canonical 20-DoF order.')
        self.config = PreTrainedConfig.from_pretrained(str(checkpoint))
        if self.config.type != 'act':
            raise ValueError('This runner requires an ACT checkpoint.')
        self.config.device = device
        self.policy = ACTPolicy.from_pretrained(str(checkpoint), config=self.config).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.config, pretrained_path=str(checkpoint),
            preprocessor_overrides={'device_processor': {'device': device}},
            postprocessor_overrides={'device_processor': {'device': 'cpu'}},
        )
        self.images = {key.removeprefix('observation.images.'): (ft.shape[1], ft.shape[2], ft.shape[0])
                       for key, ft in self.config.input_features.items() if key.startswith('observation.images.')}
        if self.config.output_features['action'].shape != (20,):
            raise ValueError('Checkpoint does not output 20 joint angles.')
        self.policy.reset()

    def predict(self, state, images):
        torch = self.torch
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (20,) or not np.isfinite(state).all():
            raise ValueError('Invalid observed robot state.')
        batch = {'observation.state': torch.from_numpy(state)}
        for name, shape in self.images.items():
            image = images[name]
            if image.shape != shape or image.dtype != np.uint8:
                raise ValueError(f'{name}: expected RGB uint8 {shape}, got {image.shape}/{image.dtype}')
            batch['observation.images.' + name] = torch.from_numpy(
                np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
        with torch.inference_mode(), torch.autocast(
                device_type=self.device, dtype=torch.float16,
                enabled=self.device == 'cuda' and self.config.use_amp):
            action = self.post(self.policy.select_action(self.pre(batch)))
        action = action.detach().cpu().float().numpy().reshape(-1)
        if action.shape != (20,) or not np.isfinite(action).all():
            raise ValueError('ACT returned invalid joint targets; no command sent.')
        return action


def check_reference(session, dataset, hand_path, arm_path):
    import yaml

    arm = yaml.safe_load(arm_path.read_text())
    reference = session['arm_reference']
    names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex']
    if [arm['joint_to_motor_map'][n] for n in names] != reference['motor_ids']:
        raise ValueError('Arm motor mapping differs from training.')
    if [arm.get('joint_signs', {}).get(n, 1) for n in names] != reference['joint_signs']:
        raise ValueError('Arm signs differ from training.')
    home = json.loads((arm_path.parent / arm['home_pose_file']).read_text())
    if not np.allclose(home['motor_rad'], reference['motor_home_rad'], rtol=0, atol=1e-5):
        raise ValueError('현재 초기 자세와 학습 때 팔 기준점이 다릅니다. 임의로 새 영점을 쓰지 않습니다.')
    for current, recorded in [(hand_path, dataset / 'collection_hand_config.yaml'),
                              (hand_path.parent / 'calibration.yaml', dataset / 'collection_calibration.yaml')]:
        if current.read_bytes() != recorded.read_bytes():
            raise ValueError(f'Hand configuration/calibration differs from training: {current}')


class CommandLimiter:
    """Use the teleop arm's damped reference, initialized at the measured pose."""
    def __init__(self, state, bounds, velocity, acceleration):
        self.q = np.asarray(state, dtype=float).copy()
        self.v = np.zeros(3)
        self.lower, self.upper = bounds
        self.velocity, self.acceleration = math.radians(velocity), math.radians(acceleration)

    def advance(self, target, dt):
        target = np.asarray(target, dtype=float)
        if target.shape != (20,) or not np.isfinite(target).all() or not 0 < dt <= 0.2:
            raise ValueError('Invalid target/control interval; command not sent.')
        count = max(1, int(np.ceil(dt * 240)))
        h = dt / count
        for _ in range(count):
            accel = np.clip(400 * (target[:3] - self.q[:3]) - 40 * self.v,
                            -self.acceleration, self.acceleration)
            velocity = np.clip(self.v + h * accel, -self.velocity, self.velocity)
            position = self.q[:3] + 0.5 * h * (self.v + velocity)
            clipped = np.clip(position, self.lower, self.upper)
            velocity[position != clipped] = 0
            self.q[:3], self.v = clipped, velocity
        self.q[3:] += np.clip(target[3:] - self.q[3:], -math.radians(120) * dt, math.radians(120) * dt)
        return self.q.copy()


def offline_observation(runtime, dataset):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    split = json.loads((dataset / 'collection_act_split.json').read_text())
    ds = LeRobotDataset(runtime.repo_id, root=dataset,
                        episodes=[split['validation_episodes'][0]], video_backend='pyav')
    sample = ds[0]
    images = {name: np.rint(sample['observation.images.' + name].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
              for name in runtime.images}
    return sample['observation.state'].numpy(), images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-path', type=Path, default=DEFAULT_POLICY)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--offline', action='store_true')
    mode.add_argument('--check-only', action='store_true')
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--camera', action='append', default=[])
    parser.add_argument('--hand-config', type=Path, default=ROOT / 'configs/orcahand-right-v1-feetech/config.yaml')
    parser.add_argument('--arm-config', type=Path, default=ROOT / 'configs/am-arm200-3dof/config.yaml')
    parser.add_argument('--arm-velocity', type=float, default=30)
    parser.add_argument('--arm-acceleration', type=float, default=90)
    args = parser.parse_args()
    if not np.isfinite([args.seconds, args.arm_velocity, args.arm_acceleration]).all() or min(
            args.seconds, args.arm_velocity, args.arm_acceleration) <= 0:
        parser.error('Duration and motion limits must be positive and finite.')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s | %(message)s')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    runtime = ACTRuntime(args.policy_path, args.dataset_root, args.device)
    if args.offline:
        state, images = offline_observation(runtime, args.dataset_root)
        action = runtime.predict(state, images)
        LOG.info('Offline inference OK: %d joints, finite targets, arm_deg=%s', len(action), np.rad2deg(action[:3]).round(2))
        return
    from orca_teleop.cameras import CameraManager, parse_camera_spec
    specs = args.camera
    if not specs:
        settings = json.loads((ROOT / 'data/collector_ui/settings.json').read_text())
        specs = [f'{name}:{settings["sources"][name]}:{shape[1]}x{shape[0]}'
                 f'{":MJPG" if name != "realsense" else ""}@{runtime.meta["fps"]}'
                 for name, shape in runtime.images.items()]
    configs = [parse_camera_spec(spec) for spec in specs]
    if len(configs) != len(runtime.images) or {c.name for c in configs} != set(runtime.images):
        raise ValueError('Camera names must exactly match the checkpoint (bottom, wrist, realsense).')
    cameras = CameraManager(configs)
    try:
        if cameras.open() != runtime.images:
            raise ValueError('Live camera dimensions differ from training.')
        cameras.ensure_live()
        state, _ = offline_observation(runtime, args.dataset_root)
        runtime.predict(state, cameras.capture())
        runtime.policy.reset()
        LOG.info('Live cameras and saved ACT processors OK: %s', runtime.images)
    finally:
        cameras.close()
    if not args.execute:
        LOG.info('점검 완료. 모터 연결 없음. 실물 추론은 --execute로 실행하세요.')
        return
    check_reference(runtime.session, args.dataset_root, args.hand_config, args.arm_config)
    from orca_teleop.am200.canonical import RobotJointCommand
    from orca_teleop.am200.hw_sink import Am200HardwareSink
    from orca_teleop.utils import RateTicker
    sink = Am200HardwareSink(hand_config_path=args.hand_config, arm_config_path=args.arm_config,
                            camera_configs=configs)
    normal_finish = False
    try:
        sink.connect()
        if sink.joint_ids != runtime.joints:
            raise ValueError('Hardware joint order differs from training.')
        if not np.allclose(sink._arm._motor_home, runtime.session['arm_reference']['motor_home_rad'], rtol=0, atol=1e-5):
            raise ValueError('Connected arm reference differs from training.')
        observation = sink.get_observation()
        limiter = CommandLimiter(observation.joint_state, sink.arm_joint_limits,
                                 args.arm_velocity, args.arm_acceleration)
        dt = 1 / runtime.meta['fps']
        ticker = RateTicker(dt)
        end = time.monotonic() + args.seconds
        step = 0
        LOG.info('실물 ACT 시작: %.1f초 / %d Hz. Ctrl+C: 중단 후 현재 팔 자세 유지.', args.seconds, runtime.meta['fps'])
        while time.monotonic() < end:
            start = time.monotonic()
            obs = sink.get_observation()
            target = runtime.predict(obs.joint_state, obs.images)
            if time.monotonic() - start > 0.2:
                raise RuntimeError('Camera/inference stalled (>200 ms); no stale command dispatched.')
            command = limiter.advance(target, dt)
            sink.dispatch_action(RobotJointCommand.now(tuple(sink.joint_ids), command))
            if step % runtime.meta['fps'] == 0:
                LOG.info('step=%d inference+read=%.1fms', step, 1000 * (time.monotonic() - start))
            step += 1
            ticker.tick()
        normal_finish = True
    except KeyboardInterrupt:
        LOG.info('중단. 추가 동작 없이 팔 유지.')
    finally:
        sink.close(return_home=normal_finish)


if __name__ == '__main__':
    main()
