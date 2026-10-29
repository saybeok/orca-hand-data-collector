"""Continuous, empirical distal-skin clearance calibration.

A confirmed physical contact supplies one signed-distance constraint, not a
3-D correspondence or an independently measured joint angle. Fit just one
clearance bias per pair. This does not identify motor transmission, grasp
force, or a preferred joint pose. Collision hulls approximate compliant skin.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .contact_calibration import FINGERS, fingerprint, pose_frames

# These are candidate-profile consistency checks, deliberately distinct from
# the existing 3 mm point-correspondence fit. Neither is a physical guarantee.
MAX_BIAS_M = .015
MAX_HELD_OUT_ERROR_M = .005


class SkinContactGeometry:
    """Signed distance/Jacobian of the five URDF distal collision-skin hulls."""

    def __init__(self, retargeter):
        import coal

        self.coal = coal
        self.r = retargeter
        self.request = coal.DistanceRequest()
        self.request.enable_signed_distance = True
        self.shapes = []
        self.mesh_hashes = {}
        self.frames = list(retargeter._frame_map.dip)
        urdf = Path(retargeter.urdf_path).resolve()
        tree = ET.parse(urdf).getroot()
        loader = coal.MeshLoader()
        for frame in self.frames:
            link = tree.find(f"link[@name='{frame}']")
            collisions = [] if link is None else link.findall('collision')
            skins = [c for c in collisions if c.find('geometry/mesh') is not None
                     and 'skin' in c.find('geometry/mesh').get('filename', '')]
            if len(skins) != 1:
                raise ValueError(f"Expected exactly one distal collision skin on {frame}")
            collision = skins[0]
            mesh = collision.find('geometry/mesh')
            uri = mesh.get('filename')
            relative = uri.removeprefix('package://')
            if uri.startswith('package://'):
                candidates = [p / relative for p in urdf.parents]
            else:
                candidates = [urdf.parent / relative]
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                raise ValueError(f"Cannot resolve collision mesh {uri}")
            scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError(f"Invalid collision mesh scale on {frame}")
            bvh = loader.load(str(path), scale)
            bvh.buildConvexHull(True, 'Qt')
            if bvh.convex is None:
                raise ValueError(f"Cannot construct collision hull for {frame}")
            origin = collision.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite(np.r_[xyz, rpy]).all():
                raise ValueError(f"Invalid collision origin on {frame}")
            placement = retargeter._pin.SE3(retargeter._pin.rpy.rpyToMatrix(rpy), xyz)
            self.shapes.append((bvh, bvh.convex, retargeter._robot.get_frame_index(frame), placement))
            self.mesh_hashes[uri] = hashlib.sha256(path.read_bytes()).hexdigest()

    def distances(self, pairs=(0, 1, 2, 3), with_jacobian=True):
        """Use the robot's last FK pose; no hardware or second FK evaluation."""
        robot = self.r._robot
        transforms = [robot.data.oMf[g[2]] * g[3] for g in self.shapes]
        transforms = [self.coal.Transform3s(t.rotation, t.translation) for t in transforms]
        distances, jacobians = [], []
        for pair in pairs:
            result = self.coal.DistanceResult()
            distance = self.coal.distance(self.shapes[0][1], transforms[0],
                                          self.shapes[pair+1][1], transforms[pair+1],
                                          self.request, result)
            if not np.isfinite(distance):
                raise ValueError("Nonfinite collision-skin distance")
            distances.append(distance)
            if with_jacobian:
                frames = [self.shapes[i][2] for i in (0, pair+1)]
                points = [result.getNearestPoint1(), result.getNearestPoint2()]
                offsets = [robot.data.oMf[i].rotation.T @ (p-robot.data.oMf[i].translation)
                           for i, p in zip(frames, points, strict=True)]
                _, jac = robot.frame_points(frames, offsets)
                derivative = np.asarray(result.normal) @ (jac[1]-jac[0])
                jacobians.append(derivative[self.r._model_q_indices_np])
        return np.asarray(distances), np.asarray(jacobians)


def fit_clearance_biases(distances):
    """Four contacts/pair fit a robust scalar bias; final two only validate."""
    if set(distances) != set(FINGERS[1:]):
        raise ValueError("Clearance calibration requires all four pairs")
    biases, residuals, before, training, loo = {}, {}, {}, {}, {}
    for finger in FINGERS[1:]:
        values = np.asarray(distances[finger], dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError(f"{finger}: exactly six finite confirmed contacts are required")
        bias = float(np.median(values[:4]))
        if abs(bias) > MAX_BIAS_M:
            raise ValueError(f"{finger}: clearance bias exceeds 15 mm")
        held_out = values[4:]-bias
        if np.max(np.abs(held_out)) > MAX_HELD_OUT_ERROR_M:
            raise ValueError(f"{finger}: held-out clearance residual exceeds 5 mm")
        if np.mean(abs(held_out)) > np.mean(abs(values[4:])) + 1e-10:
            raise ValueError(f"{finger}: calibration worsens held-out mean absolute clearance error")
        biases[finger] = bias
        residuals[finger] = held_out.tolist()
        before[finger] = values[4:].tolist()
        training[finger] = (values[:4]-bias).tolist()
        loo[finger] = [float(values[i]-np.median(np.delete(values[:4], i))) for i in range(4)]
    return dict(schema=1, method='empirical_skin_clearance', status='candidate',
                clearance_bias_m=biases, validation_signed_errors_m=residuals,
                before_validation_signed_errors_m=before, training_signed_errors_m=training,
                training_leave_one_out_errors_m=loo,
                held_out_limit_m=MAX_HELD_OUT_ERROR_M, training_samples=16, validation_samples=8,
                powered_contact_verified=False,
                limitations='Passive motor-derived angles and convex skin hulls; '
                            'not a kinematic identification or a 3 mm point-fit validation. '
                            'Side/back contact, compliance, transmission under load and grasp force are unverified.')


def fit_surface_contacts(dataset, r, geometry):
    if dataset.get('schema') != 1 or dataset.get('fingerprint') != fingerprint(r):
        raise ValueError("Contact samples are stale for the current calibration/model")
    distances = {f: [] for f in FINGERS[1:]}
    for sample in dataset.get('samples', []):
        finger = sample.get('finger')
        if finger not in distances or sample.get('operator_confirmed_contact') is not True:
            raise ValueError("Only operator-confirmed physical contacts may fit clearance biases")
        pose_frames(r, sample['angles_deg'])
        d, _ = geometry.distances([FINGERS[1:].index(finger)], with_jacobian=False)
        distances[finger].append(float(d[0]))
    profile = fit_clearance_biases(distances)
    profile.update(fingerprint=fingerprint(r), mesh_sha256=geometry.mesh_hashes,
                   surface_frames=geometry.frames,
                   measured_model_clearances_m=distances)
    return profile


class CalibratedSurfaceContacts(SkinContactGeometry):
    def __init__(self, path, retargeter):
        super().__init__(retargeter)
        profile = json.loads(Path(path).read_text())
        if (profile.get('schema') != 1 or profile.get('method') != 'empirical_skin_clearance'
                or profile.get('fingerprint') != fingerprint(retargeter)
                or profile.get('mesh_sha256') != self.mesh_hashes
                or profile.get('surface_frames') != self.frames):
            raise ValueError("Skin contact profile is incompatible or stale")
        # Recompute the consistency checks from the stored data, rather than
        # trusting a caller-provided 'validated' flag or residual summary.
        checked = fit_clearance_biases(profile.get('measured_model_clearances_m', {}))
        self.biases = np.array([checked['clearance_bias_m'][f] for f in FINGERS[1:]])
        if profile.get('clearance_bias_m') != checked['clearance_bias_m']:
            raise ValueError("Skin contact biases do not match the recorded clearance data")
        self.profile = profile
