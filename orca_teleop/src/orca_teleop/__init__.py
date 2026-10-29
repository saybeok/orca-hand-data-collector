"""Public API loaded on demand; camera/UI tools need no motor or torch imports."""
from importlib import import_module

_EXPORTS = {
    **{name: 'cameras' for name in ('CameraManager', 'OpenCVCamera', 'OpenCVCameraConfig',
                                  'list_available_cameras', 'parse_camera_spec')},
    **{name: 'ingress.server' for name in ('HandLandmarks', 'IngressServer')},
    **{name: 'pipeline' for name in ('OrcaHandSink', 'RecordableSink', 'SinkObservation',
                                   'TeleopQueues', 'retargeter_worker', 'robot_worker',
                                   'run', 'run_local', 'run_manus_local')},
    'Retargeter': 'retargeting.retargeter',
}
__all__ = ['pipeline', *_EXPORTS]


def __getattr__(name):
    if name == 'pipeline':
        value = import_module('.pipeline', __name__)
    elif name in _EXPORTS:
        value = getattr(import_module('.' + _EXPORTS[name], __name__), name)
    else:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    globals()[name] = value
    return value
