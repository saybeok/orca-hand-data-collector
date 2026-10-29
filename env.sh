# source this before running any ORCA teleop / control command.

# ~/.bashrc sources ROS 2 Humble and CUDA, which leak into the 3.11 venv:
# PYTHONPATH shadows venv packages such as lark, and LD_LIBRARY_PATH makes the
# loader bind ROS's libeigenpy.so instead of the cmeel-bundled one, which
# breaks `import pinocchio` with an undefined EIGENPY_ARRAY_API symbol.
unset PYTHONPATH
unset LD_LIBRARY_PATH

export ORCA_HAND_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Retargeter resolves the URDF as
# $ORCAHAND_DESCRIPTION_DIR/v1/models/urdf/orcahand_<hand>.urdf
export ORCAHAND_DESCRIPTION_DIR="$ORCA_HAND_ROOT/orcahand_description"

# v1 right hand with Feetech motors; the bundled orca_core config assumes Dynamixel.
# OrcaHand takes the config.yaml itself, not its directory.
export ORCA_MODEL_PATH="$ORCA_HAND_ROOT/configs/orcahand-right-v1-feetech/config.yaml"
export ORCA_URDF_PATH="$ORCAHAND_DESCRIPTION_DIR/v1/models/urdf/orcahand_right.urdf"
export ORCA_RETARGET_CONFIG="$ORCA_HAND_ROOT/configs/adaptive_analytical_orca_v1.yaml"

# Confirmed camera roles and USB-port selectors shared by preview and recording.
source "$ORCA_HAND_ROOT/configs/cameras.env"
