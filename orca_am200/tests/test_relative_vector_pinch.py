"""AKO-inspired hand-only priorities, not proof of physical pad contact."""
import numpy as np
import pytest
import yaml

from test_pinch_retarget import (
    AdaptiveAnalyticalRetargeter, HAND_CONFIG, ROOT, URDF, _free_thumb_pose,
)

PROFILE = ROOT/'configs/adaptive_analytical_orca_v1_pinch_calibrated.yaml'


@pytest.fixture
def robot():
    return AdaptiveAnalyticalRetargeter(hand_config_path=str(HAND_CONFIG),
        urdf_path=str(URDF), config_path=str(PROFILE))


def args(r, kp):
    return (r._compute_tip_vectors(kp), r._compute_tip_dirs(kp),
            r._compute_full_hand_vectors(kp), r._compute_pinch_alpha(kp),
            r._neutral_qpos_phys, r._compute_pinch_vectors(kp), r._thumb_ip_target_bend(kp))


def pose(r, gap, pair=0):
    kp = _free_thumb_pose(r, 53)
    kp[8+4*pair] = kp[4] + [gap/100, 0, 0]
    return kp


def test_collection_uses_one_point_definition_and_no_hull_or_taught_pose(robot):
    r = robot
    assert r._pinch_objective == 'relative_vector'
    assert r._long_finger_tracking == 'bone_directions'
    assert r._surface_contacts is None and r._taught_contacts is None
    assert r._frame_offsets is r._contact_offsets
    assert not np.any(r._frame_offsets)
    assert not np.any(r._surface_contact_closure_margin_cm)
    assert r._pinch_target_mode == 'landmark'
    np.testing.assert_array_equal(r._pinch_pair_weights, [8]*4)


@pytest.mark.parametrize('pair', range(4))
def test_closed_pinch_retains_three_dimensional_position_and_analytic_gradient(robot, pair):
    r = robot; kp = pose(r, 0, pair)
    r._w_pos = r._w_dir = r._w_full_hand = r._w_thumb_ip_bend = r._norm_delta = 0
    values = args(r, kp)
    q = r._neutral_qpos_phys + .1
    loss, analytic = r._loss_and_grad(q, *values)
    assert loss > 0 and np.linalg.norm(analytic) > 0
    eps = 1e-6
    numeric = np.array([(r._loss_and_grad(q+d, *values)[0]-r._loss_and_grad(q-d, *values)[0])/(2*eps)
                        for d in np.eye(16)*eps])
    np.testing.assert_allclose(analytic, numeric, rtol=2e-4, atol=1e-5)
    inactive = [i for i,j in enumerate(r._finger_joint_ids)
                if not j.startswith(('thumb_', ('index','middle','ring','pinky')[pair]+'_'))]
    np.testing.assert_allclose(analytic[inactive], 0, atol=1e-12)
    # Equal target *length* with a different sideways direction must have a
    # different cost. A scalar nearest-hull clearance cannot distinguish it.
    pos, _ = r._robot.compute_positions_and_jacobians(r._full_model_qpos(q),
        r._computed_frame_indices, r._contact_offsets)
    vector = (pos[r._tip_indices[pair+1]]-pos[r._tip_indices[0]])*100
    aligned, opposite = list(values), list(values)
    aligned[5] = values[5].copy(); opposite[5] = values[5].copy()
    aligned[5][pair] = vector; opposite[5][pair] = -vector
    assert r._loss_and_grad(q, *opposite)[0] > r._loss_and_grad(q, *aligned)[0] + 1


@pytest.mark.parametrize('gap', [0, 3.5, 5.1])
def test_complete_objective_gradient_through_shape_release(robot, gap):
    r = robot; kp = pose(r, gap, 2); values = args(r, kp)
    q = r._neutral_qpos_phys + .1
    _, analytic = r._loss_and_grad(q, *values)
    eps = 1e-6
    numeric = np.array([(r._loss_and_grad(q+d, *values)[0]-r._loss_and_grad(q-d, *values)[0])/(2*eps)
                        for d in np.eye(16)*eps])
    np.testing.assert_allclose(analytic, numeric, rtol=2e-4, atol=1e-5)


def test_closed_pair_releases_all_conflicting_shape_targets_but_keeps_direction(robot):
    r = robot; kp = pose(r, 0); values = list(args(r, kp))
    q = r._neutral_qpos_phys + .1
    loss, grad = r._loss_and_grad(q, *values)
    changed = list(values)
    changed[0] = values[0].copy(); changed[2] = values[2].copy()
    changed[0][[0,1]] += 20
    changed[2][0] += 20  # Thumb proximal is not part of the new term.
    changed[2][1] *= 2  # Proximal length is free, its live direction remains.
    for offset in (5,10): changed[2][[offset,offset+1]] += 20
    new_loss, new_grad = r._loss_and_grad(q, *changed)
    assert new_loss == pytest.approx(loss, abs=1e-9)
    np.testing.assert_allclose(new_grad, grad, atol=1e-9)
    # The distal direction objective is still live for both closing fingers.
    changed = list(values); changed[1] = values[1].copy()
    changed[1][0] *= -1
    assert r._loss_and_grad(q, *changed)[0] != pytest.approx(loss)
    changed = list(values); changed[2] = values[2].copy(); changed[2][1] *= -1
    assert r._loss_and_grad(q, *changed)[0] != pytest.approx(loss)
    # An uninvolved finger must still follow its chain posture.
    changed = list(values); changed[2] = values[2].copy(); changed[2][3] += 20
    assert r._loss_and_grad(q, *changed)[0] > loss


def test_approach_release_have_no_entry_memory_or_ip_switch_and_hold_fixed_joints(robot):
    r = robot
    fixed = {j: float(np.rad2deg(r._neutral_qpos_phys[i])) for i,j in enumerate(r._finger_joint_ids)
             if j.startswith(('middle_', 'ring_', 'pinky_'))}
    r.set_fixed_hand_joints(fixed)
    previous = r._neutral_qpos_phys.copy()
    for gap in list(np.linspace(6,0,35)) + list(np.linspace(0,6,35)):
        kp = pose(r, gap)
        q = r._solve(kp, dt_s=1/30)
        assert np.max(abs(q-previous)) <= np.deg2rad(2)+1e-8
        assert np.all(q >= r._lower_phys) and np.all(q <= r._upper_phys)
        assert np.isnan(r._pinch_entry_vectors_cm).all()
        np.testing.assert_allclose(r._last_solve_pinch_vectors, r._compute_pinch_vectors(kp), atol=1e-12)
        for j,v in fixed.items(): assert np.rad2deg(q[r._finger_joint_ids.index(j)]) == pytest.approx(v)
        previous = q.copy()
    seed = r._neutral_qpos_phys.copy(); results=[]
    for gap in (5, 5-1e-5):
        r.reset(); r._last_qpos_phys = seed.copy()
        results.append(r._solve(pose(r,gap))[r._thumb_ip_index])
    np.testing.assert_allclose(results, results[0], atol=1e-7)


def test_trace_reports_vector_error_without_claiming_skin_contact(robot):
    r=robot; q=r._solve(pose(r,0)); previous=q.copy()
    cmd=dict(zip(r._finger_joint_ids, np.rad2deg(q)))
    trace=r.command_diagnostics(cmd,cmd,123)
    assert trace['pinch_objective'] == 'relative_vector'
    assert trace['powered_contact_verified'] is False
    assert 'target_clearance_mm' not in trace
    assert trace['target_tip_gap_mm'][0] == pytest.approx(0)
    assert trace['final_tip_vector_error_mm'][0] == pytest.approx(trace['final_model_tip_gap_mm'][0])
    np.testing.assert_array_equal(r._last_qpos_phys,previous)


def test_bend_diagnostic_reports_input_and_model_angles_without_changing_solver_state(robot,caplog):
    r=robot;kp=pose(r,0,2);q=r._solve(kp);before=r._last_qpos_phys.copy()
    with caplog.at_level('INFO'):
        r._log_pinch_diagnostic(kp,np.rad2deg(q))
    text='\n'.join(record.message for record in caplog.records)
    assert 'Finger bend diag' in text and 'order=index,middle,ring,pinky' in text
    assert 'input_proximal_distal_angle_deg=' in text
    assert 'model_proximal_distal_angle_deg=' in text
    assert 'pinch_proximal_dir_weight=20' in text
    np.testing.assert_array_equal(r._last_qpos_phys,before)


def operator_from_robot(r, q):
    """Different human lengths, same achievable long-finger directions."""
    pos,_=r._robot.compute_positions_and_jacobians(r._full_model_qpos(q),
        r._computed_frame_indices,r._contact_offsets)
    kp=np.zeros((21,3))
    for f in range(5):
        b,p,d,t=pos[[r._base_indices[f],r._pip_indices[f],r._dip_indices[f],r._tip_indices[f]]]
        if f == 0:
            kp[1:5]=np.array([b,p,d,t])+[.5,0,0]  # No accidental pinch.
        else:
            proximal=(p-b)/np.linalg.norm(p-b)
            distal=(t-d)/np.linalg.norm(t-d)
            start=1+4*f
            kp[start]=b
            kp[start+1]=b+.03*proximal
            kp[start+2]=kp[start+1]+.025*distal
            kp[start+3]=kp[start+2]+.02*distal
    return kp,pos


def test_free_shape_uses_own_base_and_robot_lengths_including_coincident_v1_frames(robot):
    r=robot
    q=r._neutral_qpos_phys.copy()
    for i,j in enumerate(r._finger_joint_ids):
        if not j.startswith('thumb_'):
            q[i]=np.deg2rad(25 if j.endswith('mcp') else 35 if j.endswith('pip') else 0)
    kp,pos=operator_from_robot(r,q)
    targets=r._compute_full_hand_vectors(kp).reshape(3,5,3)[:,1:]/100
    expected=np.stack([pos[idx][1:]-pos[r._base_indices][1:]
                       for idx in (r._pip_indices,r._dip_indices,r._tip_indices)])
    np.testing.assert_allclose(targets,expected,atol=1e-10)
    np.testing.assert_allclose(targets[0],targets[1],atol=1e-12)
    for f in range(1,5):
        indices=slice(1+4*f,5+4*f)
        # Changing an operator's finger length or palm width must not
        # manufacture a curl: directions, not absolute positions, transfer.
        kp[indices]=kp[indices]*(1+.2*f)+[.02*f,-.03,.01]
    np.testing.assert_allclose(r._compute_full_hand_vectors(kp).reshape(3,5,3)[:,1:]/100,
                               expected,atol=1e-10)
    np.testing.assert_allclose(r._compute_tip_vectors(kp)[1:]/100,expected[-1],atol=1e-10)


def test_free_long_finger_direction_is_live_without_changing_thumb_direction_rule(robot):
    r=robot;kp=pose(r,8)
    assert not r._compute_pinch_alpha(kp).any()
    r._w_pos=r._w_full_hand=r._w_thumb_ip_bend=r._norm_delta=0
    q=r._neutral_qpos_phys+.1
    loss,grad=r._loss_and_grad(q,*args(r,kp))
    assert loss>0
    for finger in ('index','middle','ring','pinky'):
        assert np.linalg.norm(grad[[i for i,j in enumerate(r._finger_joint_ids) if j.startswith(finger+'_')]])>0
    np.testing.assert_allclose(grad[:4],0,atol=1e-10)


@pytest.mark.parametrize('pair',range(4))
def test_new_free_tracking_does_not_change_the_fully_closed_pair_objective(robot,pair):
    r=robot
    r._w_pinch_proximal_dir=0  # Isolate free-shape tracking from the new pinch term.
    finger=('index','middle','ring','pinky')[pair]
    r.set_fixed_hand_joints({j:float(np.rad2deg(r._neutral_qpos_phys[i]))
                            for i,j in enumerate(r._finger_joint_ids)
                            if not j.startswith(('thumb_',finger+'_'))})
    kp=pose(r,0,pair);q=r._neutral_qpos_phys+.1
    new_loss,new_grad=r._loss_and_grad(q,*args(r,kp))
    r._long_finger_tracking='palm_position'
    old_loss,old_grad=r._loss_and_grad(q,*args(r,kp))
    assert new_loss==pytest.approx(old_loss,abs=1e-9)
    np.testing.assert_allclose(new_grad,old_grad,atol=1e-9)


@pytest.mark.parametrize('pair', range(4))
def test_pinch_proximal_direction_has_correct_gradient_and_does_not_target_thumb(robot,pair):
    r=robot;kp=pose(r,0,pair);q=r._neutral_qpos_phys+.1
    r._w_pos=r._w_dir=r._w_full_hand=r._w_thumb_ip_bend=r._norm_delta=0
    values=list(args(r,kp));values[5]=None  # Isolate proximal direction from tip closure.
    loss,grad=r._loss_and_grad(q,*values)
    assert loss>0 and np.linalg.norm(grad)>0
    eps=1e-6
    numeric=np.array([(r._loss_and_grad(q+d,*values)[0]-r._loss_and_grad(q-d,*values)[0])/(2*eps)
                      for d in np.eye(16)*eps])
    np.testing.assert_allclose(grad,numeric,rtol=2e-4,atol=1e-6)
    finger=('index','middle','ring','pinky')[pair]
    unrelated=[i for i,j in enumerate(r._finger_joint_ids) if not j.startswith(finger+'_')]
    np.testing.assert_allclose(grad[unrelated],0,atol=1e-10)


@pytest.mark.parametrize('condition',['open','disabled','fixed','pinch_off'])
def test_proximal_pinch_term_cannot_change_free_motion_or_disabled_pairs(robot,condition):
    r=robot
    kp=pose(r,8 if condition=='open' else 0,2)
    if condition=='disabled':r._pinch_pair_weights[2]=0
    if condition=='fixed':
        r.set_fixed_hand_joints({j:float(np.rad2deg(r._neutral_qpos_phys[i]))
                                for i,j in enumerate(r._finger_joint_ids) if j.startswith('ring_')})
    if condition=='pinch_off':r._w_pinch=0
    q=r._neutral_qpos_phys+.1;values=args(r,kp)
    loss,grad=r._loss_and_grad(q,*values)
    r._w_pinch_proximal_dir=0
    old_loss,old_grad=r._loss_and_grad(q,*values)
    assert loss==pytest.approx(old_loss,abs=1e-10)
    np.testing.assert_allclose(grad,old_grad,atol=1e-10)


def test_four_fingers_can_open_close_and_reopen_without_latching_ring_curl(robot):
    r=robot
    r.set_fixed_hand_joints({j:float(np.rad2deg(r._neutral_qpos_phys[i]))
                            for i,j in enumerate(r._finger_joint_ids) if j.startswith('thumb_')})
    previous=r._neutral_qpos_phys.copy()
    for mcp,pip in ((5,0),(65,70),(5,0)):
        reference=r._neutral_qpos_phys.copy()
        for i,j in enumerate(r._finger_joint_ids):
            if not j.startswith('thumb_'):
                reference[i]=np.deg2rad(mcp if j.endswith('mcp') else pip if j.endswith('pip') else 0)
        kp,_=operator_from_robot(r,reference)
        for _ in range(85):
            q=r._solve(kp,dt_s=1/30)
            assert np.max(abs(q-previous))<=np.deg2rad(2)+1e-8
            previous=q.copy()
        np.testing.assert_allclose(q[4:],reference[4:],atol=np.deg2rad(1))


@pytest.mark.parametrize('override', [dict(pinch_objective='typo'), dict(w_pinch=0),
    dict(w_pinch_proximal_dir=-1),dict(w_pinch_proximal_dir=float('nan')),
    dict(w_pinch_proximal_dir=float('inf')),dict(long_finger_tracking='palm_position'),
    dict(long_finger_tracking='typo'),dict(pinch_objective='adaptive'),
    dict(pinch_target_mode='entry_anchored'),dict(surface_contact_calibration='unused.json'),
    dict(contact_calibration='unused.json'),dict(contact_pose_library='unused.json')])
def test_conflicting_objective_configuration_is_rejected(tmp_path, override):
    cfg=yaml.safe_load(PROFILE.read_text());cfg['retarget'].update(override)
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        AdaptiveAnalyticalRetargeter(hand_config_path=str(HAND_CONFIG),urdf_path=str(URDF),config_path=str(path))
