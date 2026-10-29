"""Empirical pinch targets: actual stored samples, bounded motion, free release.

The fixture contains synthetic configurations, not asserted physical contacts.
These tests check command behavior only, never real contact or grasp success.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from orca_teleop.retargeting.adaptive_analytical import AdaptiveAnalyticalRetargeter
from orca_teleop.retargeting.contact_calibration import FINGERS, TaughtContacts, fingerprint

ROOT = Path(__file__).resolve().parents[2]


def build(config):
    return AdaptiveAnalyticalRetargeter(
        hand_config_path=str(ROOT/'configs/orcahand-right-v1-feetech/config.yaml'),
        urdf_path=str(ROOT/'orcahand_description/v1/models/urdf/orcahand_right.urdf'),
        config_path=str(config),
    )


@pytest.fixture
def profile(tmp_path):
    r = build(ROOT/'configs/adaptive_analytical_orca_v1_pinch_refined.yaml')
    samples = []
    for f in FINGERS[1:]:
        for n in range(6):
            q = r._neutral_qpos_phys.copy()
            idx = [i for i,j in enumerate(r._finger_joint_ids) if j.startswith(('thumb_', f+'_'))]
            q[idx] = r._lower_phys[idx] + (.30+.06*n)*(r._upper_phys[idx]-r._lower_phys[idx])
            samples.append(dict(finger=f, operator_confirmed_contact=True,
                                angles_deg=dict(zip(r._finger_joint_ids, np.rad2deg(q).tolist()))))
    path = tmp_path/'library.json'
    path.write_text(json.dumps(dict(schema=1, fingerprint=fingerprint(r), samples=samples)))
    config = r.config
    config['retarget']['contact_pose_library'] = path.name
    cfg = tmp_path/'taught.yaml'; cfg.write_text(yaml.safe_dump(config))
    return cfg, path


def keypoints(finger=None, gap_cm=0):
    kp = np.zeros((21,3))
    kp[1:5] = [[.02,0,.03],[.02,0,.06],[.01,0,.09],[0,0,.12]]
    for n in range(4):
        x = .07*(n+1)
        kp[5+4*n:9+4*n] = [[x,.02,.04],[x,.02,.07],[x,.02,.10],[x,.02,.13]]
    if finger is not None:
        i = FINGERS[1:].index(finger)
        kp[8+4*i] = kp[4] + [gap_cm/100,0,0]
    return kp


@pytest.mark.parametrize('finger', FINGERS[1:])
def test_each_pair_reaches_one_recorded_pose_and_releases_continuously(profile, finger):
    cfg, _ = profile
    r = build(cfg)
    previous = r._neutral_qpos_phys.copy()
    for _ in range(70):
        q = r._solve(keypoints(finger))
        assert np.max(abs(q-previous)) <= np.deg2rad(2)+1e-8
        assert np.all(q >= r._lower_phys) and np.all(q <= r._upper_phys)
        previous = q.copy()
    target = r._last_taught_target
    assert target.finger == finger and target.strength == 1
    np.testing.assert_allclose(q[target.indices], target.angles, atol=1e-9)
    # Only the pair's seven variables are constrained; other joints retain
    # nonzero objective gradients and are solved by the original retargeter.
    assert len(target.indices) == 7
    opened = r._solve(keypoints())
    assert r._last_taught_target is None
    assert np.max(abs(opened-previous)) <= np.deg2rad(2)+1e-8
    assert np.max(abs(opened[target.indices]-previous[target.indices])) > 1e-5


def test_pose_selection_uses_preshape_and_stays_latched_until_release(profile):
    cfg,_ = profile; r=build(cfg); lib=r._taught_contacts
    args=(r._pinch_d1,r._pinch_d2,r._pinch_pair_weights)
    q=lib.poses['ring'][0].copy()
    a=lib.target(keypoints('ring',4),q,*args)
    assert a.pose == 0 and 0 < a.strength < 1
    b=lib.target(keypoints('ring'),lib.poses['ring'][5],*args)
    assert b.pose == 0
    assert lib.target(keypoints(),q,*args) is None
    c=lib.target(keypoints('ring'),lib.poses['ring'][5],*args)
    assert c.pose == 5


def test_simultaneous_pinches_and_disabled_pairs_use_base_controller(profile):
    cfg,_=profile; r=build(cfg); lib=r._taught_contacts
    kp=keypoints('middle');kp[16]=kp[4]
    assert lib.target(kp,r._neutral_qpos_phys,r._pinch_d1,r._pinch_d2,r._pinch_pair_weights) is None
    enabled=np.ones(4);enabled[1]=0
    assert lib.target(keypoints('middle'),r._neutral_qpos_phys,r._pinch_d1,r._pinch_d2,enabled) is None


def test_free_motion_is_numerically_identical_to_the_working_profile(profile):
    cfg,_=profile
    taught=build(cfg)
    base=build(ROOT/'configs/adaptive_analytical_orca_v1_pinch_refined.yaml')
    for _ in range(5):
        np.testing.assert_allclose(taught._solve(keypoints()),base._solve(keypoints()),atol=1e-10)


def test_nonparticipating_index_keeps_moving_during_taught_middle_contact(profile):
    cfg,_=profile;r=build(cfg);kp=keypoints('middle')
    for _ in range(70):
        before=r._solve(kp)
    target=r._last_taught_target
    kp[6:9]=[[-.02,.04,.07],[-.03,.05,.06],[-.04,.06,.05]]
    for _ in range(40):
        after=r._solve(kp)
    assert r._last_taught_target.finger == 'middle'
    np.testing.assert_allclose(after[target.indices],target.angles,atol=1e-9)
    index=[i for i,j in enumerate(r._finger_joint_ids) if j.startswith('index_')]
    assert np.max(abs(after[index]-before[index])) > np.deg2rad(5)


@pytest.mark.parametrize('hz',[15,30,60])
def test_recorded_goal_obeys_time_bound_on_approach_and_release(profile,hz):
    cfg,_=profile;r=build(cfg);prev=r._neutral_qpos_phys.copy()
    for kp in (keypoints('pinky'),keypoints()):
        total=0
        for _ in range(hz):
            q=r._solve(kp,dt_s=1/hz)
            delta=np.max(abs(q-prev));total+=delta
            assert delta <= np.deg2rad(60/hz)+1e-8
            prev=q.copy()
        assert total > np.deg2rad(5)


def test_taught_approach_gradient_matches_finite_difference(profile):
    cfg,_=profile;r=build(cfg);kp=keypoints('index',3.5)
    target=r._taught_contacts.target(kp,r._neutral_qpos_phys,r._pinch_d1,r._pinch_d2,r._pinch_pair_weights)
    args=(r._compute_tip_vectors(kp),r._compute_tip_dirs(kp),r._compute_full_hand_vectors(kp),
          r._compute_pinch_alpha(kp),None,r._compute_pinch_vectors(kp),r._human_thumb_ip_bend(kp),target)
    q=r._neutral_qpos_phys+.1;_,grad=r._loss_and_grad(q,*args)
    eps=1e-6
    numeric=np.array([(r._loss_and_grad(q+d,*args)[0]-r._loss_and_grad(q-d,*args)[0])/(2*eps)
                      for d in np.eye(len(q))*eps])
    np.testing.assert_allclose(grad,numeric,rtol=1e-5,atol=1e-5)


def test_recalibration_or_unconfirmed_pose_is_rejected(profile):
    cfg,path=profile;r=build(cfg)
    data=json.loads(path.read_text());data['samples'][0]['operator_confirmed_contact']=False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='confirmed'):
        TaughtContacts(path,r)
    data['samples'][0]['operator_confirmed_contact']=True;data['fingerprint']='stale'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='stale'):
        TaughtContacts(path,r)


def test_reset_forgets_contact_selection(profile):
    cfg,_=profile;r=build(cfg);r._solve(keypoints('middle'))
    assert r._last_taught_target is not None
    r.reset()
    assert r._last_taught_target is None and r._taught_contacts._finger is None
