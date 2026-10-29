# ORCA Hand v1 (Feetech, right) setup

Teleoperation stack for a v1 ORCA Hand with Feetech HLS servos, driven from a
USB webcam through MediaPipe.

## Layout

```
orca_core/               hardware control, tensioning, calibration (editable install)
orcahand_description/    URDF/MJCF models
orca_teleop/             retargeting pipeline, owns the .venv used by everything
configs/
  orcahand-right-v1-feetech/config.yaml   hand config (Feetech variant of the v1 right hand)
  adaptive_analytical_orca_v1.yaml        retarget config (v1 joint names)
scripts/                 wrappers that source env.sh and pass the right paths
env.sh                   environment for every command
```

## Setup notes

### Ubuntu desktop collector

```bash
./scripts/collector_ui.sh
```

The desktop app provides three camera previews, editable USB camera assignments,
Quest/robot connection diagnostics, arm command speed/acceleration settings, a
success counter and the dataset folder. Connect teleop first, then use **녹화
시작**, **성공 저장**, **실패 · 재시도**, and **세션 종료**. Only saved trials count
towards the target (default 50). Stop/closing the app discards the unfinished
trial and finalizes previously saved episodes. Wait for finalization before
disconnecting hardware. Unlike the terminal collector, this app does **not**
automatically move home between trials: reposition through teleop, then explicitly
start the next recording. The camera preview stays available during recording
through the recorder, without opening each USB camera twice.

Settings and session logs are in `data/collector_ui/`; real LeRobot datasets are
in `data/lerobot/`. Each dataset also records the hand calibration/configuration,
joint ordering and connect-time arm motor reference in `collection_session.json`
and `collection_*` files. The arm reference is preserved for later coordinate
conversion; it is **not** a globally fixed mechanical zero or a completed π0.5
training adapter. Upload/training remain separate steps.

`./scripts/collector_ui.sh --demo` tests UI/record/discard/save without opening
robot buses or cameras. It writes dummy hand-only state/action, **not** a real
20-DoF demonstration, under `data/collector_demo/`.

To move to another Ubuntu PC, export the **modified working files**, including
motor mapping, calibration, contact profile and meshes:

```bash
./scripts/export_collector.sh
```

Copy the printed archive to the new PC, extract it into a new `~/orca_hand`
folder, install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
cd ~/orca_hand
./scripts/install_collector_ubuntu.sh
```

The installer creates a fresh Python environment, sets up Quest USB permissions
and adds **ORCA 시연 수집** to the application menu. Log out/in for USB group
permissions. Camera USB paths may change: select the new inputs in the app and
verify the three views. The two HD cameras have duplicate serials, so explicitly
assign them to USB ports. Do not copy `.venv` from the old PC. The export excludes
virtualenvs, git metadata and recorded datasets; copy existing datasets separately
if needed. Installing on a second PC does not require recalibrating the same hand.

### Physical camera selection on Linux

Camera numbers are logical video inputs and may change when devices reconnect.
One RealSense can expose depth/IR, RGB and metadata nodes. For this setup use
the physical USB port selectors for the RGB inputs:

```bash
cd /home/lee/orca_hand
source env.sh
orca_teleop/.venv/bin/python orca_teleop/scripts/preview_cameras.py \
  --camera realsense:usb=2-1.2:640x480@30 \
  --camera bottom:usb=1-1.1:640x480@30 \
  --camera wrist:usb=1-1.3:640x480@30
```

Close any earlier preview with `q` before reopening the cameras. USB selectors
resolve at open time, prefer the RGB interface, and exclude metadata streams.
Linux capture opens an explicit `/dev/videoN` with the V4L2 backend, without
falling back to another backend. Duplicate USB selectors under different camera
names are rejected. The same `--camera` specs work with `record_dataset.py`.
These USB paths describe the current cabling, not permanent serial identities;
if a cable/hub port changes, run `--list-cameras` again. Both HD cameras report
the same serial, so their serial alone cannot distinguish them. The USB path
identifies which physical connection is intended; verify the views before
assigning final front/wrist/side feature names.

For focus inspection, close earlier previews and open separate full-frame
windows instead of the reduced grid. The command prints the controls exposed
by each camera without setting focus or exposure:

```bash
source env.sh
orca_teleop/.venv/bin/python orca_teleop/scripts/preview_cameras.py --inspect-controls \
  --camera "$ORCA_CAMERA_BOTTOM" \
  --camera "$ORCA_CAMERA_WRIST" \
  --camera "$ORCA_CAMERA_REALSENSE"
```

Startup logs distinguish `captured_shape` (the actual camera frame) from
`output_shape` (the configured/resized frame). Neither resizing nor enlarging
the preview can recover optical detail absent in the capture. The control
listing is needed before assuming the HD cameras have electronic autofocus.
Both HD inputs use an explicit `:MJPG` pixel-format suffix in `cameras.env`:
`bottom:usb=1-1.1:640x480:MJPG@30`, for example. This reduces shared USB2
bandwidth; frames returned to the recorder remain decoded RGB arrays. A
rejected format is an error rather than a silent uncompressed fallback.
The RealSense retains its driver-selected capture format. `--arrange` opens
separate 640x480 windows in two columns for comparing focus without the grid's
additional image reduction.

The operator confirmed the input then labelled `video3` as the **bottom view**
on 2026-10-28. Current cabling selects `usb=1-1.4`, saved in
`configs/cameras.env` and loaded by `source env.sh` as `ORCA_CAMERA_BOTTOM`.
Pass `--camera "$ORCA_CAMERA_BOTTOM"` to either preview or recording; the
recorded image feature is `observation.images.bottom`. The operator then
confirmed USB `1-1.3` as the **wrist view**, saved as `ORCA_CAMERA_WRIST` with
image feature `observation.images.wrist`. `ORCA_CAMERA_REALSENSE` retains the
device name, with image feature `observation.images.realsense`. It uses
`serial=261643069030` to survive USB port and video node changes. The identical
HD cameras share a serial and must retain their assigned USB ports; changing
ports requires confirming their roles again. Missing or duplicate serials fail
explicitly rather than selecting a different camera.
The earlier `ORCA_CAMERA_HD_UNASSIGNED` variable remains an alias of the wrist
spec for existing commands. Reload `source env.sh` after this mapping update.
For a labelled comparison:

```bash
source env.sh
orca_teleop/.venv/bin/python orca_teleop/scripts/preview_cameras.py --grid \
  --camera "$ORCA_CAMERA_BOTTOM" \
  --camera "$ORCA_CAMERA_WRIST" \
  --camera "$ORCA_CAMERA_REALSENSE"
```

For real Hwatu collection, close the previews and other teleop processes, then
run `./scripts/record_hwatu.sh`. It uses Quest, the current continuous calibrated
pinch profile, left-hand pinch arm clutch, arm scale 0.35, and all three cameras
at 640x480/30 Hz. The default is one manually ended episode, written to a new
`data/lerobot/hwatu_<timestamp>_<pid>` directory without uploading. The state and
action vectors have 20 entries; hand state is derived from motor encoders.
Connect both robot USB adapters and Quest (authorize USB debugging), then open
`http://localhost:8765` in Quest Browser and start hand tracking. Collection starts
when teleop and observations are ready. SPACE saves the episode; q/Esc saves the
current episode and finishes (it does **not** discard a failed attempt).
`./scripts/record_hwatu.sh --check-only` checks USB paths and cameras without
connecting to motors. Add `--num-episodes 5` for a longer session; from episode 2
the existing recorder automatically returns the robot home and rests before
recording again.

The real teleop and recording shell wrappers limit arm commands to 15 degrees/s
per joint and 45 degrees/s² acceleration, using the existing continuous motion
reference before dispatch and recording. Override with
`--arm-max-velocity-deg-s 10 --arm-max-acceleration-deg-s2 30` for a slower trial.
These are teleop command limits, not measured motor speed guarantees; the
between-episode homing routine has its own interpolation.

If Quest reports `insufficient permissions` even though the user belongs to
`plugdev`, install the included rule for the observed Quest 3 USB ID `2833:5013`:

```bash
sudo install -m 0644 configs/udev/70-orca-quest.rules /etc/udev/rules.d/70-orca-quest.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --action=add --subsystem-match=usb --attr-match=idVendor=2833 --attr-match=idProduct=5013
adb kill-server
adb start-server
adb devices -l
```

If needed, unplug and reconnect only the Quest cable. Accept USB debugging in
the headset if it shows `unauthorized`; continue when `adb devices` shows
`device`. Run ADB and recording as the normal user, not with `sudo`.

To identify every readable logical input visually, close earlier previews with
`q`, then run:

```bash
orca_teleop/.venv/bin/python orca_teleop/scripts/preview_cameras.py --all
```

This displays a grid labelled `cam<number>`, video node and USB path. Inputs
that cannot reopen together are shown as unavailable while the others continue.
Automatic preview requests 640x480 at 15 fps to reduce USB bandwidth; this does
not change recording settings. `--grid` also combines explicitly selected
`--camera` views into one window. Press `q` or Escape to close the preview.

### Upstream DexPilot comparison (ORCA v1 right)

`--dexpilot` selects the unmodified five-finger `DexPilotOptimizer` from
[dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting), pinned
to 0.5.0. The existing adaptive profiles remain the default and comparison.
Install this optional backend alongside the extras you already use, for example
`uv sync --extra adaptive --extra metaquest --extra dexpilot` from `orca_teleop/`.
For LeRobot recording, use `uv sync --extra learning --extra dexpilot`.
The current project virtualenv already has DexPilot installed.

```bash
cd /home/lee/orca_hand
# Stop other teleop processes first. Quest input; only the physical hand connects.
./scripts/teleop_am200_hw.sh --no-arm --dexpilot
# Optional simulator comparison:
./scripts/teleop_am200.sh --no-arm --dexpilot
# Return to the working adaptive candidate:
./scripts/teleop_am200_hw.sh --no-arm --pinch-refined
```

`configs/dexpilot_orca_v1.yaml` uses upstream projection/release thresholds
3/5 cm, uniform scale 1.0, default LP alpha 0.1, and URDF tip-frame origins.
There is no adaptive auto-scale, per-finger boost, extra pad offset, or added
joint-angle correction. The ORCA adapter aligns the canonical 21-point input
to the URDF frame, optimizes only 16 finger joints within the hardware config
ROM, and passes the wrist through the existing signed-degree convention.
Neutral warm-start and filter initialization, complete reset of projection
history, a separate 60 deg/s elapsed-time output bound, and conversion to the existing canonical command are local integration
choices. Off-hand reset clears solver/filter history; it does not calibrate
hand size. The global size ratio can be set with `scaling_factor` in the YAML.
`execution.max_joint_velocity_deg_s: 0` disables the local command bound for
offline comparison; it is not a DexPilot objective parameter.

This is the library's generalized DexPilot implementation, not an exact
reproduction of the original four-finger paper. Its discrete projection can
still feel like snapping and its smoothing can lag. It does not provide full
mesh collision avoidance or guaranteed physical pad contact. Hardware grasp
quality still needs operator comparison. Do not combine `--dexpilot` with
adaptive `--pinch*` or `--key-objectives` flags.

Replay the same saved Quest input without USB:

```bash
source env.sh
orca_teleop/.venv/bin/python orca_teleop/scripts/eval_retarget_pinch.py \
  data/hand_captures/pinch_01.npz --retargeter dexpilot \
  --retarget-config configs/dexpilot_orca_v1.yaml \
  --output-dir data/eval/dexpilot
```

For recording, select `--retargeter dexpilot --retarget-config
/home/lee/orca_hand/configs/dexpilot_orca_v1.yaml`. Use the same frozen profile
for an entire dataset.

### Working adaptive pinch: selective refinement trial

The known working `--pinch` profile remains available. `--pinch-refined` keeps
the same coordinates, offsets, auto-scale, ROM and 60 deg/s command bound.
It retains `pinch_pair_weights` (index 1, middle/ring/pinky 2; missing weights
default to 1), and enables `pinch_relax_tip_position`. This releases the
conflicting palm-relative tip-position objective as pinch activation grows:
its original coefficient is multiplied by `1 - smoothstep(activation)`.
The same rule applies to all four pairs, with the thumb using the strongest
enabled pair. Fingertip direction, full-chain posture, temporal regularization
and the contact loss remain unchanged. There is no extra curl, joint-angle
offset or forced joint stop. Index/middle output can also change during pinches.
With no active pair, or with relaxation disabled (the default), the original
objective remains unchanged. Set `pinch_relax_tip_position: false` in this YAML
to disable position relaxation independently of the thumb-bend preference.

The refined profile uses `thumb_ip_tracking: direct_free` for standalone thumb
bending. With no active enabled pinch pair, the thumb DIP optimization variable
is fixed to the measured input bend minus the URDF rest bend, clipped to the
physical ROM and the existing elapsed-time velocity bounds. The remaining
variables are solved normally; there is no second optimizer or post-solve
motor overwrite. Invalid thumb segments hold that variable at its prior value.
During an active pinch the fixed-variable rule is disabled, retaining the
preceding soft thumb-bend/contact/posture objective. Incoming poses can still
differ after free tracking; identical pinch inputs from identical starting
states produce identical solutions.

The input bend is the angle between landmarks 2->3 and 3->4. The robot's signed
bone bend is `thumb_dip + rest_bend`; the v1 rest bend (about 7.71 degrees) is
derived from URDF geometry and hinge direction. The `w_thumb_ip_bend: 5.0`
soft objective remains active during pinches with `thumb_ip_bend_scope: always`.
For older profiles `thumb_ip_tracking` defaults to `soft` and bend scope defaults
to `pinch`. Use `thumb_ip_tracking: soft` to compare the preceding always-soft
tracking. No motor ID, calibration, or physical ROM is changed.

Three human points provide flexion magnitude, not signed hyperextension; bad
tracking can still give the wrong target. Degenerate thumb segments skip this
term. `input_thumb_ip_bend_deg` and `model_thumb_ip_bend_deg` in `Pinch diag`
compare input magnitude with the signed model bone angle. A negative **joint**
angle alone does not necessarily imply a negative **bone** bend. This is a
local objective extension, not an unchanged official or paper implementation.
Set `thumb_ip_tracking: soft` and `w_thumb_ip_bend: 0` to compare the preceding
position-relaxation profile.
To reproduce the old weight-only profile, use soft tracking, zero bend weight,
and disable position relaxation.
Saved soft pinch-only traces are in `data/eval/thumb_ip_bend/`; the preceding
always-soft replay is in `data/eval/thumb_ip_follow/`.
In that single-capture comparison, median free-motion model bend error changes
from 26.6 to 1.6 degrees, with similar or smaller median pinch-frame tip gaps.
A controlled free-thumb segment rotation test also follows 20/40-degree inputs
at about 19/39 degrees after settling. Those results describe the preceding
always-soft mode, not the current direct-free mode. Direct-free tests separately
check 0/20/40/60/0-degree settled tracking, frame velocity bounds, invalid input
holding, identical pinch solves given the same initial pose, and command
conversion through the canonical adapter and calibration routing to motor 15.
Use soft tracking with `thumb_ip_bend_scope: pinch` to reproduce the older scope.
Physical pad alignment and actual card grasps have not been validated.

To distinguish hand tracking, command generation, and hardware following, run
`./scripts/teleop_am200_hw.sh --no-arm --pinch-refined --hand-feedback-debug`.
The current refined startup log should include `scope=free`. `Pinch diag` reports input and
model thumb bone bend and the DIP command. `Hand feedback` reports the applied
DIP command and the motor-encoder-derived joint estimate once per second,
using the existing writer thread and serial connection. Failed cached reads
are rejected. Compare while holding a bend, since an immediate read can lag a
moving command. Encoder agreement does not prove tendon/link-angle agreement.
This diagnostic is opt-in; normal teleop adds no feedback reads.

If the physical thumb end joint stays still, stop teleop with Ctrl+C and run
`./scripts/hand.sh test-thumb`. This bypasses Quest and retargeting and opens
only the configured `thumb_dip` motor (ID 15 on this hand). It moves 40 joint
degrees from the measured starting pose at 15 degrees/s, holds for 3 seconds,
and returns, within the existing ROM. It prints command and motor-derived
angle estimates, then releases that motor. No arm/wrist commands, calibration
sweep, neutral pose, or configuration/calibration writes are performed.
Observe which physical joint actually moves: matching encoder estimates alone
cannot verify the motor-to-joint mapping or tendon transmission.
The default 40-degree move covers only part of the configured 132-degree IP
ROM. If calibration visibly reaches both ends but that probe looks small,
`./scripts/hand.sh test-thumb --delta-deg 90` requests a larger round trip at
the same speed. Targets remain clamped to the existing ROM; the calibration,
current limit and teleoperation mapping are unchanged.

Add `--visualize-landmarks` to the hardware teleop command to open the incoming
21-point skeleton before retargeting. The red thumb and the `RAW thumb MCP` /
`IP (end joint)` readout show bends calculated directly from those points.
This distinguishes bending at the thumb base from bending at its end joint;
it does not measure the human's physical angle independently. The option uses
the existing landmark viewer and does not change robot commands.

For model/physical pad alignment, a separate full-hand contact calibration is
available. This retains the working adaptive solver and fits effective local
contact points for **all five fingers together**, rather than adding fixed curl
angles or increasing pinch weights. It requires confirmed physical contacts;
non-contact teleop logs cannot supply those measurements.

```bash
# Stop teleop first. Releases only the 16 finger motors, never wrist/arm.
./scripts/hand.sh contacts capture
# Offline: writes a separate JSON profile and retarget YAML only if checks pass.
./scripts/hand.sh contacts fit
./scripts/teleop_am200_hw.sh --no-arm \
  --retarget-config "$PWD/configs/adaptive_analytical_orca_v1_contact_calibrated.yaml" \
  --hand-feedback-debug
```

Support the hand and gently place the actual gripping pads in contact. Do not
force a joint or backdrive a stiff motor. Record six **different bends** for
each thumb–index/middle/ring/pinky pair; moving only the wrist supplies no useful
variation. Capture never enables torque, sends position goals, changes current
limits, or runs neutral/calibration. It reads stable motor-derived angles after
the operator confirms contact. It saves every accepted pose and resumes the
same dataset on the next run. These angles are not independent link sensing;
slack/backlash can make this procedure unsuitable and cause validation to fail.

Four poses per pair fit the points; the final two are held out. Poorly
identified fits, corrections over 15 mm, validation residuals over 3 mm, or a
pair whose mean validation error worsens are rejected. Those are local
engineering acceptance thresholds, not official ORCA or paper parameters.
Passing means consistency of the recorded contact constraints, not proven
physical contact under powered teleoperation. Check additional real poses
afterwards. A fixed offset cannot repair pose-dependent tendon transmission.

Records default to `data/contact_calibration/`. Existing motor calibration,
ROM, mapping, and the refined profile are not rewritten. The generated profile
is bound to the motor calibration, mapping, ROM, fingertip frames, and URDF;
changes refuse stale profiles. After changing tendon tension, recalibrate the
motors and collect new contacts. For a new session use, for example,
`--samples data/contact_calibration/contacts_v2.json` on **both** commands.

For a **continuous** alternative to the failed point fit, the existing 24
records can calibrate an empirical distal-skin clearance target:

```bash
# Offline; reuses the existing recording and preserves motor calibration.
./scripts/hand.sh contacts fit-surface
./scripts/teleop_am200_hw.sh --no-arm --pinch-calibrated --hand-feedback-debug
# Compare the preceding continuous point objective:
./scripts/teleop_am200_hw.sh --no-arm --pinch-refined --hand-feedback-debug
```

`--pinch-calibrated` selects the separate
`configs/adaptive_analytical_orca_v1_pinch_calibrated.yaml`. The backend remains
`adaptive_analytical`. As a pair closes, its point-vector objective blends into
a signed convex-skin clearance objective **inside the same optimizer**. Its
Jacobian follows the two closest skin points, including their angular motion.
Shape/direction objectives, free thumb tracking, physical ROM and the existing
60 degrees/s bound remain. There is no recorded-pose selection, joint-angle
overwrite, or extra fixed bend. With no active pinch, the output matches the
refined profile numerically. A disabled pair receives no surface objective.

Confirmed contact establishes one scalar clearance constraint, rather than
requiring arbitrary fingertip origins to coincide in 3-D. For each pair the
median of the **first four** measured-model clearances supplies one bias; the
final two are held out. The candidate refuses biases over 15 mm, held-out
absolute errors over **5 mm**, or worsening mean absolute error for any pair.
This is a separate empirical-distance criterion: it does **not** relax, pass,
or replace the existing 3 mm point-fit validation. The profile includes every
training residual and leave-one-out residual; the small held-out check does
not establish accuracy throughout the hand workspace. It is fingerprinted
against motor calibration, ROM, mapping, URDF and all five collision meshes.

For the current recording, held-out mean absolute **model clearance** errors
are index 5.44→2.57 mm, middle 1.61→1.61 mm, ring 8.16→2.94 mm and pinky
2.94→2.94 mm; the largest held-out residual is 4.81 mm. Some training/leave-one-out
residuals still reach about 7 mm. This limited consistency result is why the
profile remains a **candidate**, not an established full-hand calibration.
The ring bias is about +10 mm: the nominal meshes may remain visibly separated
when the real assembly's manually recorded pads were touching. Conversely,
some recorded contacts overlap the nominal hulls. Motor-angle estimates with
torque off do not identify the loaded transmission or compliant skin. Closest
skin points need not be the desired opposing pad faces. Actual powered contact,
side/back contact, grasp stability, and force remain unverified. Check the real
hand before using this candidate for demonstrations.

`Skin contact diag` reports commanded-model signed clearances, calibrated
targets and residuals in mm. These are not tactile or independent joint-angle
measurements. `Pinch diag` continues to report tip-origin distances separately;
those origins need not coincide with this surface objective. The original
profiles, 24 raw records and motor limits are retained for comparison.

The current calibrated YAML uses per-pair `surface_contact_closure_margin_mm`:
index and middle **8 mm**, ring and pinky **10 mm**, following the operator's
report that ring/pinky still under-closed at 8 mm. Scalar values remain valid
and apply the same margin to every pair; a mapping must name all four pairs.
At full pinch these values request that much less signed clearance relative
to the fitted empirical bias. The extra closure fades smoothly to zero as that pair
opens; open-hand tracking is unchanged. This is an explicit runtime tuning,
separate from the recorded contacts and fitted biases. It is not a newly
measured calibration or a guarantee on physical motion. The original
candidate is retained as
`configs/adaptive_analytical_orca_v1_pinch_calibrated_no_margin.yaml`; setting
the margin to `0` also restores its targets. The prior 2 mm tuning is retained
as `configs/adaptive_analytical_orca_v1_pinch_calibrated_margin2.yaml`.
The preceding all-8-mm tuning is retained as
`configs/adaptive_analytical_orca_v1_pinch_calibrated_margin8.yaml`.
The accepted range is 0–10 mm; physical joint ROM is unchanged.
Startup logs report the active margin, and `Skin contact diag` includes it in
the reported target clearance. Regenerating the YAML with `contacts fit-surface`
uses the base settings; reapply this explicit tuning afterwards if desired.

The 24 recorded poses can also be used directly as empirical pinch targets,
without declaring the failed geometric fit successful:

```bash
# Offline: snapshots the existing recording and generates the taught profile.
./scripts/hand.sh contacts teach
# Quest, physical hand only:
./scripts/teleop_am200_hw.sh --no-arm --pinch-taught --hand-feedback-debug
```

`--pinch-taught` selects `configs/adaptive_analytical_orca_v1_pinch_taught.yaml`.
Open-hand tracking is identical to the working refined profile. On entering a
pinch, the controller selects the closest recorded configuration for the thumb
and participating finger, using joint distances normalized by each ROM. It
latches that sample until release so contact targets do not hop during a grasp.
The approach uses a smooth distance-dependent pose objective. At full pinch
(the existing scaled 2 cm threshold), only those seven joint variables move to
the selected measured configuration inside the existing 60 degrees/s and ROM
bounds; the remaining fingers continue through the base optimizer. The nominal
FK pair target blends toward the selected sample's recorded-model vector,
which can be nonzero even though the operator confirmed actual pad contact.
Opening beyond the existing 5 cm threshold releases the taught target and
allows a different one of the six poses to be selected on the next approach.
Simultaneous full multi-finger pinches fall back to the base retargeter, since
independent pair recordings do not establish a valid combined contact pose.

This constrains the closed pinch posture; it is not unrestricted continuous
retargeting inside a closed grasp, a kinematic identification, or a tactile
controller. No grasp force or actual powered pad contact has been measured.
Passive captured angles may differ under powered tendon load. Consequently
the command replay checks are not a physical success-rate evaluation. The
original strict `contacts fit` rejection and its acceptance thresholds remain
in force. The taught snapshot is fingerprinted against motor calibration and
the robot model, and does not change current limits, mapping, or motor limits.
The simulator can use `--pinch-taught` to inspect commands, but its nominal
meshes need not touch at these physical calibration configurations.

`Pinch diag` includes `taught_contact=middle:pose3:blend1.00`, for example,
and `taught_joint_error_deg` is the maximum **command** difference from that
recorded pose. It is not an independent link-angle/contact measurement. Pair
`target_gap_cm` reports the target actually used by the optimizer, including
the recorded-model vector. Motor-derived following errors are still printed
by `--hand-feedback-debug`.

```bash
./scripts/teleop_am200.sh --no-arm --pinch-refined
# Explicit physical hand-only trial, after stopping any other teleop:
./scripts/teleop_am200_hw.sh --no-arm --pinch-refined
```

The candidate YAML is `configs/adaptive_analytical_orca_v1_pinch_refined.yaml`.
Older weight-only traces are in `data/eval/adaptive_refinement/`; objective
ablations and before/after traces are in `data/eval/adaptive_objectives/`.
Held-pose ablations identify palm-relative tip position as a competing objective.
The moving replay has mixed results: ring gap median improves, while pinky
whole-trajectory median does not. Longer held pinches close better; this does
not solve short-duration tracking. Measured gaps are URDF frame distances, not
verified physical pad contact or collision-free grasps. Real-hand performance
has not been validated; return to `--pinch` to compare.

Reproduce the objective comparison without connecting either robot bus:

```bash
source env.sh
orca_teleop/.venv/bin/python orca_teleop/scripts/eval_retarget_pinch.py \
  data/hand_captures/pinch_01.npz \
  --retarget-config configs/adaptive_analytical_orca_v1_pinch_refined.yaml \
  --variant baseline:pinch_relax_tip_position=0,w_thumb_ip_bend=0 \
  --variant relaxed:w_thumb_ip_bend=0 \
  --output-dir data/eval/adaptive_objectives
```

### Experimental key-objectives hand backend

`--key-objectives` explicitly selects an independent hand optimizer based on
[Xin et al., Analyzing Key Objectives in Human-to-Robot Retargeting](https://mingrui-yu.github.io/retargeting/),
DOI `10.1109/RAP.2026.3656110`. Existing adaptive defaults, hardware arm control,
motor mapping and calibration files remain the comparison baseline.

From this repository root, try simulation first:

```bash
./scripts/teleop_am200.sh --no-arm --key-objectives
# Optional card scene with existing left-hand arm clutch:
./scripts/teleop_am200.sh --key-objectives --task hwatu_mat --clutch --arm-scale 0.35
```

The implementation is `orca_teleop/src/orca_teleop/retargeting/key_objectives.py`,
configured by `configs/key_objectives_orca_v1.yaml`. It combines squared
wrist-relative fingertip, DIP-to-tip, thumb-relative pinch, joint-prior and
previous-command losses. Shape/pinch use complementary sigmoid weights;
pinch distance uses the paper's continuous piecewise-linear rescaling. All
objectives use the same URDF fingertip frame origins, with no extra virtual-tip
offsets. SI metres/radians are used throughout.

This is an ORCA **hand-only adaptation**, not a complete paper reproduction:
global thumb-position/wrist-orientation and arm-hand joint optimization are
omitted; the fifth-finger pair, neutral-based ORCA priors, and 30-frame uniform
scale calibration are local choices. Prior weights are ORCA/SI-specific rather
than copied LEAP joint-index values. Frame coincidence is not pad-contact or
grasp validation. A separate local execution profile restricts joint command
changes to 60 deg/s using input timestamps; it is not part of the paper's
objective. `execution.max_joint_velocity_deg_s: 0` disables that bound for
offline ablations. Do not infer physical motor-speed limits from this value.

Replay a saved Quest capture without connecting hardware:

```bash
source env.sh
orca_teleop/.venv/bin/python orca_teleop/scripts/eval_retarget_pinch.py \
  data/hand_captures/pinch_01.npz --retargeter key_objectives \
  --retarget-config configs/key_objectives_orca_v1.yaml \
  --output-dir data/eval/key_objectives
```

Hardware selection is also explicit (`teleop_am200_hw.sh --no-arm
--key-objectives`), but this backend has not been validated on the physical
hand. For future recording, `record_dataset.py` accepts `--retargeter
key_objectives --retarget-config /absolute/path/to/key_objectives_orca_v1.yaml`.
Freeze and archive the chosen configuration when collecting a dataset; the
runtime hand-size calibration is still session-dependent. Do not mix the
adaptive `--pinch`/`--pinch-legacy` profile with this backend.

### Local environment

Everything runs from the single virtualenv at `orca_teleop/.venv` (Python 3.11).
`orca_core` is installed there as an editable checkout, so the calibration
scripts and the teleop pipeline always agree on one version.

Three things about this machine required deviating from the upstream READMEs:

- **`~/.bashrc` sources ROS 2 Humble and CUDA.** `PYTHONPATH` then shadows venv
  packages, and `LD_LIBRARY_PATH` makes the loader bind ROS's `libeigenpy.so`
  instead of the cmeel-bundled one, which breaks `import pinocchio` with an
  undefined `EIGENPY_ARRAY_API` symbol. `env.sh` unsets both.
- **`orca_core` uses this repository's editable `../orca_core` checkout.**
  Both `pyproject.toml` and the lockfile now use that source, so `uv sync`
  retains the local Feetech conventions used by the calibration files.
- **Meshcat is not in any extra.** The error message suggests
  `--extra urdf-viz`, which does not exist; `uv pip install meshcat` is the fix.

## Hardware config

The bundled `orca_core/models/v1/orcahand-right/config.yaml` assumes Dynamixel
motors at 3M baud. The copy in `configs/` changes only the transport:

| field | value | why |
| --- | --- | --- |
| `motor_type` | `feetech` | selects `FeetechClient` |
| `baudrate` | `1000000` | the only rate in `MOTOR_BAUD_RATES[feetech]` |
| `control_mode` | `current_based_position` | Feetech honours mode 5 as a torque limit |
| `max_current` | `400` | mA maps onto the 0-1000 torque register, so 40% |

Joint maps, ROMs, neutral pose and the calibration sequence are unchanged from
the v1 hand.

This hand's adapter is a QinHeng `1a86:55d3`, a CH343/CH9102 that binds
`cdc-acm` and therefore appears as **`/dev/ttyACM0`**, not the `/dev/ttyUSB0`
you get from the older CH340 chips. `port: auto` resolves it either way, since
orca_core matches on the USB vendor ID rather than the device name.

Motor layout, derived from `joint_to_motor_map` (sign stripped):

| | ID | model |
| --- | --- | --- |
| wrist | 17 | HLS3930 |
| fingers | 16..1 | HLS3915 |

The highest ID attaches directly to the board, then each lower ID chains off the
one above it: board -> 17 -> 16 -> ... -> 1.

## Retarget config

`joint_name_aliases` upstream renames `thumb_pip` to `thumb_cmc`, which only
fits the v2 joint vocabulary. The v1 URDF names that joint `right_thumb_pip`,
exactly as the v1 hand config does, so `configs/adaptive_analytical_orca_v1.yaml`
empties the alias map. The frame block needs no change: the default
`robot_frames` in the upstream config is already the v1 layout.

`orca_teleop/src/orca_teleop/urdf_viz.py` had the same alias hardcoded; it now
picks whichever of the two names the URDF actually defines, so it works for both
v1 and v2.

## Running

```bash
scripts/smoke_retarget.py   # offline: synthetic hand, no camera or hardware
scripts/teleop_viz.sh       # webcam -> Meshcat viewer at http://127.0.0.1:7000/static/
```

Bring-up, in order, with the hand plugged in and powered:

```bash
scripts/hand.sh detect      # confirm the port and which motors answer
scripts/hand.sh configure   # one-time: burn IDs/baud into a fresh motor chain
scripts/hand.sh tension     # hold taut, tension tendons by hand, Ctrl+C when done
scripts/hand.sh calibrate   # 28-step limit sweep, writes calibration.yaml
scripts/hand.sh neutral     # move to the neutral pose
scripts/teleop_hw.sh        # webcam -> physical hand
```

To adjust just the thumb IP tendon, stop teleop and run
`scripts/hand.sh tension --joints thumb_dip --no-move-motors`.
Only the selected joint's motor (ID 15 on this hand) is opened and held at its
freshly measured position. Adjust its tendon tension manually, then Ctrl+C
releases that motor. Omit `--no-move-motors` to include the usual bidirectional
winding routine, still restricted to the selected motor. The command holds the
motor for manual adjustment; it does not mechanically loosen the tensioner.
After changing tendon tension, run
`scripts/hand.sh calibrate --joints thumb_dip`.

If a Feetech calibration records almost identical endpoints (for example
6.0868 and 6.0899 rad), do not use that ratio for teleop. A sweep now fails if
its span is no larger than twice the hardstop position tolerance, and retains
the prior calibration. Selected-motor setup also rejects measured positions
outside the driver's writable single-turn frame instead of silently clamping
both motion directions to the same endpoint.
For encoder-frame recovery, stop motor control, place the affected joint near
the middle of its travel with torque released, and run
`scripts/hand.sh recenter --motors 15 --apply`. This changes that motor's stored
encoder offset. Only after it reports a native position near 2048, run
`scripts/hand.sh calibrate --joints thumb_dip` again. Recalibration is required
after an encoder-offset change; old motor limits cannot be reused.

`configure` is interactive: it prompts for one motor at a time and, because
Feetech needs an unpowered hotplug, asks you to unplug and replug USB between
motors. Motors that already carry their target ID are skipped, so it is safe to
re-run after an interruption. Skip it entirely if the chain is already
programmed — `detect` will list the configured IDs.

Add `--mock` to `tension`, `calibrate` or `neutral` for a dry run without
hardware.

Hold your hand still in frame at the start of either teleop command: the
retargeter emits nothing until it has auto-scaled, which takes about 30 frames.
