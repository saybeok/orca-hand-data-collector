# ORCA Hand v1 + AM-ARM200: teleoperation and data collection

Quest WebXR teleoperation and three-camera LeRobot collection for the physical
3-DoF AM-ARM200 arm and 17-DoF right ORCA Hand v1 Feetech. Webcam/MediaPipe and
MuJoCo entry points are also included.

## 새 PC 설치 (현재 수집 구성, 2026-11-12)

수집 전용 PC에서는 [orca-hand-data-collector](https://github.com/saybeok/orca-hand-data-collector)의
한국어 설치 안내를 사용하세요. 이 레포에도 동일한 최신 제어·리타겟팅·수집 코드를 보관합니다.

```bash
git clone https://github.com/saybeok/orca-hand-data-collector.git
cd orca-hand-data-collector
# uv 설치 후 실행: https://docs.astral.sh/uv/getting-started/installation/
./scripts/install_collector_ubuntu.sh
# 로그아웃/로그인 후
./scripts/collector_ui.sh
```

설치기는 Python 3.11과 잠금 파일의 의존성을 설치하고 장치 없이 모델/GUI/인코더를
점검합니다. 같은 로봇의 저장된 팔 초기 자세를 처음 한 번 복원하며, 이미 저장된 자세는
덮어쓰지 않습니다. 기본 수집 지시는 `move the bottle.`, 팔 명령 속도/가속도는
30°/s·90°/s²입니다. 기존 PC의 저장된 설정이 있으면 그 설정을 우선합니다.
새 PC에서는 카메라 세 역할과 저장 폴더를 앱에서 다시 선택하세요. `.venv`는 복사하지 않습니다.

설치만 확인하려면 `./scripts/install_collector_ubuntu.sh --check-only`를 실행하세요.
모터·카메라를 열거나 움직이지 않습니다. 수집 데이터와 학습 체크포인트는 Git에 넣지 않습니다.
공개 수집본은 [Hugging Face](https://huggingface.co/datasets/saybeok03/orca-hand-bottle)에
있고, `collection-v2-165`는 165개 에피소드·141,740프레임의 고정 버전입니다.

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
disconnecting hardware. The app uses the saved arm initial pose for recording start, successful save,
failed retry and session close. Return motions are outside the recorded trial;
release and re-engage the left-hand clutch after a return. The camera preview stays available during recording
through the recorder, without opening each USB camera twice.

The app starts maximized, with a large RealSense view and two supporting views.
Click a camera to make it the main view; double-click to open a separate
enlargement. F11 toggles application fullscreen and Escape exits it. The
equal-size option shows three side-by-side views. Previews fit their actual
available area, preserve aspect ratio, and keep the recording feed's full
640x480 resolution. Collection settings, devices and logs live in sidebar tabs;
record/save/retry/stop stay visible below the videos.

Settings and session logs are in `data/collector_ui/`; real LeRobot datasets are
under `data/lerobot/`. Recording saves JPEG-95 capture sessions in a separate writer process,
with bounded local IPC and acknowledged durable saves. This keeps JPEG work
outside the teleoperation interpreter. Video encoding and image statistics
run after collection.
After ending collection, click **LeRobot 일괄 변환** and select a session or its
parent directory. Verified LeRobot output appears in each session's `lerobot/`
subdirectory; capture sources remain available for retries. CLI conversion:
`orca_teleop/.venv/bin/python orca_teleop/scripts/convert_recordings.py --source data/lerobot`.
Only the converted directory is ready for Hugging Face upload/training.
Live arm IK consumes the newest Quest wrist pose independently of finger
optimization. Single-slot control mailboxes retain recalibration requests and
brief clutch-release events while replacing old poses. The separate dataset
queue still records at 30 Hz, using the last successfully dispatched complete
20-DoF command, not a command waiting to be sent. Restart the teleop session
after updating. `Arm control | ... input-to-command ...` in the session log
reports PC-side input-to-command computation time, not headset-to-motor latency.
On hardware connection, the arm's servo position limits are read and converted
into the taught-home coordinate frame before binding IK. This prevents an
unreachable servo goal from accumulating as a fictitious arm pose. The firmware
limits and saved home are not rewritten. A home at a servo endpoint still has
no travel in that direction; teach an interior pose if both directions are needed.
With torque off, the measured arm can fall outside the servo's allowed command
range. Startup rejects deviations greater than 1 degree before staging a goal
or enabling torque, and reports each affected motor and its deviation. Support
the arm and manually restore it near the saved initial pose before reconnecting;
do not widen this tolerance to bypass a large displacement.
This unit's shoulder-pan polarity was changed from +1 to -1 on 2026-11-08
after the operator confirmed opposite lateral motion. Existing taught homes
with the opposite polarity can be updated
explicitly with `orca_teleop/.venv/bin/python scripts/migrate_arm_pan_home.py --apply`.
The helper backs up the home and changes only its pan-polarity metadata; it
opens no serial devices and preserves the stored motor positions.
Each dataset also records the hand calibration/configuration,
joint ordering and connect-time arm motor reference in `collection_session.json`
and `collection_*` files. The arm reference is preserved for later coordinate
conversion; it is **not** a globally fixed mechanical zero or a completed π0.5
training adapter. Upload/training remain separate steps.

`./scripts/collector_ui.sh --demo` tests UI/record/discard/save without opening
robot buses or cameras. It writes dummy hand-only state/action, **not** a real
20-DoF demonstration, under `data/collector_demo/`.

To move to another Ubuntu PC, export the **modified working files**, including
motor mapping, calibration, saved arm initial pose, contact profile and meshes:

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

### Local ACT baseline (bottle demonstrations, 2026-11-08)

The local training entry point uses the installed LeRobot 0.4.4 ACT trainer
([official ACT guide](https://huggingface.co/docs/lerobot/act)):

```bash
cd /home/lee/orca_hand
bash scripts/train_act_local.sh
```

This explicitly selects the three sessions in
`configs/act_bottle_20261108_sources.json`: 14 + 4 + 12 saved episodes after the
11.1-second capture was excluded. It preserves source captures and builds one
derived dataset at `data/act/bottle_20261108/capture/lerobot/`. Compatible camera
features, joint order, units, motor calibration and arm reference are required.
JPEG/joint payloads are hard-linked to avoid duplicating raw files on disk;
the derived capture is immutable. Use a new destination for changed sources.
The preparation manifest maps every derived episode to its source session and
source episode. Failed retries and `data/excluded_recordings/` are not selected.

The recorded task is `move the bottle.`. Inputs are bottom, wrist and RealSense
RGB images at 640x480 plus the measured 20-DoF state. Targets are the recorded
20-DoF commands in canonical model radians at 30 Hz. No joint sign, zero,
retargeting or camera mapping is changed for training.

A deterministic split reserves one episode from each session: 27 train and 3
held out, listed in `collection_act_preparation.json` / `collection_act_split.json`.
Numeric normalization uses only the 27 training episodes; the original complete
statistics are kept in `meta/stats_all_episodes.json`. Images use fixed ImageNet
normalization. Held-out episodes are not used for gradient updates. This is an
episode split from the same collection, not a claim of unseen-scene performance.

The RTX 3060 6GB preset keeps the standard ResNet-18 / 512-wide ACT, CVAE,
100-step action chunk, learning rate 1e-5 and KL weight 10. It uses FP16,
batch size 4, 2 loader workers, 20,000 updates and checkpoints every 5,000 updates.
`n_action_steps=10` configures later inference to replan after ten commands;
training still predicts the complete 100-step chunk. Command-line arguments
override the preset, for example `--steps=100 --save_freq=100` for a short run.

Outputs go to a new `outputs/train/act_bottle_<timestamp>/` directory, overridable
with `ORCA_ACT_OUTPUT`. The final model is under
`checkpoints/last/pretrained_model/`; optimizer/RNG state is saved alongside it.
Resume with the installed `lerobot-train` command and
`--config_path=<run>/checkpoints/last/pretrained_model/train_config.json --resume=true`
using the same `ACCELERATE_MIXED_PRECISION=fp16` setting. The launcher disables
Hub upload, telemetry and W&B. ResNet-18 ImageNet weights must already be cached
for offline training. It opens no cameras or motor ports; deployment on the
physical robot is a separate operation.

### Run the trained ACT on the physical robot

As of 2026-11-09, new training runs follow `data/act/active_dataset.json`.
The active `bottle_20261108_min20/capture/lerobot/` dataset excludes recordings
shorter than 20 seconds: 29 episodes remain, with the same 27 training episodes
and 2 held-out episodes. The excluded 19.2-second episode was held out in the
completed run, so it did not contribute training gradients. Failed retries
were already discarded by the recorder. The original 30-episode dataset is
retained for checkpoint reproducibility; it is not an additional collection.
The completed checkpoint's inference default continues using that original
snapshot and its saved normalizers. No retraining or upload is part of curation.
Raw payloads and unchanged video streams are shared without re-encoding;
curated metadata/data rows expose only retained episodes. The curation manifest
records source indices and the recovery location.

The 20,000-step checkpoint from the local run is the default for this wrapper:

```bash
cd /home/lee/orca_hand
# Check saved model/processors against a held-out frame; opens no hardware.
bash scripts/run_act_hw.sh --offline
# Check the three live cameras and inference; opens no motor ports.
bash scripts/run_act_hw.sh --check-only
# Close collection/teleop first, then run one physical 30-second rollout.
bash scripts/run_act_hw.sh --execute --seconds 30
```

No Quest or retargeter is used: live RGB + measured joint positions feed ACT,
and saved checkpoint processors unnormalize its 20 canonical-radian targets.
The runner loads the checkpoint's saved ImageNet/numeric normalizers, rather
than rebuilding them from dataset statistics. Camera USB selectors come from
the collector's `data/collector_ui/settings.json`; repeat `--camera` to override
all three explicitly. Camera names/resolutions must match training.

The taught arm zero, signs, motor IDs and hand configuration/calibration must
match the training snapshot. Connection uses the existing slow taught-home
startup, then arm target shaping uses 30 deg/s and 90 deg/s² by default, matching
collection; hand target increments are limited to 120 deg/s. These limit
commands, not independently measured link speeds. ACT retains its trained
100-step prediction chunks and replans after ten commands, at the recorded
30 Hz control rate. These checks do not establish physical task success.

After the requested duration, the arm returns to its taught pose. Ctrl+C or an
inference/read error stops dispatch and closes while preserving the arm hold,
without a return motion. `--policy-path` and `--dataset-root` select another
compatible local checkpoint/dataset. The default command is check-only; this
code never executes physical motion without `--execute`.

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
derived from URDF geometry and hinge direction. The `w_thumb_ip_bend: 50.0`
soft objective remains active during pinches with `thumb_ip_bend_scope: always`.
For older profiles `thumb_ip_tracking` defaults to `soft` and bend scope defaults
to `pinch`. Use `thumb_ip_tracking: soft` to compare the preceding always-soft
tracking. No motor ID, calibration, or physical ROM is changed.

The collection profile uses `thumb_ip_bend_gain: 2.0` for positive thumb-IP
travel in both free motion and pinch. At a constant 53-degree input the bend
target stays about 98.3 degrees (90.6 degrees of joint travel). Previously the
free-only 2x gain faded to 1x near contact, requesting extension down to a
53-degree bone bend during approach. The constant gain removes that changing
target without changing contact margins or joint limits.
The thumb-index approach range is now 5 to 3 cm in auto-scaled landmark
coordinates (previously 6 to 3 cm); the other pairs retain 5 to 2 cm.
This is continuous target/weight blending, not DexPilot's on/off hysteresis.
An explicit legacy `thumb_ip_free_bend_gain` still overrides free-motion gain
only; profiles with neither key keep gain 1. Contact remains a soft IK goal,
and the physical grasp must be checked after reconnecting.

The collection profile now selects `pinch_target_mode: entry_anchored`. When
each enabled pair enters its approach range, the previous emitted robot pose
provides both its thumb-to-tip vector and signed skin clearance. A smoothstep
of human distance from d2 to d1 then shrinks that vector toward zero and moves
the clearance toward the existing calibrated bias minus closure margin. This
avoids interpreting a human landmark distance as a robot skin gap that can
request reopening an already-close robot at onset. The entry is frozen during
holds and reversals, released at d2, and cleared on retargeter reset/fixed-hand
mode changes. Objective evaluation and diagnostics use the same frozen targets.
With direct-free thumb tracking in this mode, the IP feasible interval widens
continuously with pinch activation, from the tracked angle to the existing
per-frame joint bounds. It no longer switches from fixed to fully free at the
first nonzero activation. The bend objective remains active throughout.
Other profiles retain the default `landmark` mapping. The final contact target,
thumb gain, joint limits and velocity bound are unchanged. This is a local
reference-generation change, not a paper implementation or sensor feedback;
it does not guarantee monotone total IK motion or physical pad contact.

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

The collector and `--pinch-calibrated` now use an **AKO-inspired relative-vector
objective inside `adaptive_analytical`**. Despite its historical filename,
`configs/adaptive_analytical_orca_v1_pinch_calibrated.yaml` no longer loads the
passive surface fit. Thumb-to-finger vectors remain active through full pinch,
instead of fading into a nearest-hull distance. Shape, distal direction,
scaling and pinch share the same URDF fingertip origins. Palm-relative tip and
chain posture goals fade smoothly for the participating fingers; distal
direction, temporal regularization, thumb IP tracking and ROM/velocity bounds
remain. All four pairs use the same priority. Folded fingers remain fixed
inside the solver and have no pinch objective. No recorded pose is selected.

The current profile explicitly records `long_finger_tracking: bone_directions`.
Index/middle/ring/pinky shape targets copy the operator's segment directions
onto the robot's own segment lengths, measured from each **current MCP frame**
with its Jacobian. In v1 the PIP/DIP frame origins coincide; the corresponding
zero-length segment is preserved. This avoids forcing one robot hinge towards
two different human joint positions, or curling a straight robot finger to
reach an underscaled palm-relative target. Distal-direction tracking stays
active in free motion for these four fingers. Thumb targets, pinch scaling,
thresholds and thumb IP gain remain unchanged. Older configs without this key
retain palm-relative shape tracking. Reconnect to load the new profile; no
hardware calibration is needed for this software change.

During a pinch, `w_pinch_proximal_dir: 20.0` retains the participating long
finger's MCP-to-PIP direction alongside its distal direction. Matching only
the distal direction allowed MCP over-flexion with PIP straightening while
the nominal tip still reached the thumb. This soft directional term adds no
fixed joint angle, pose replay or angle floor. It is zero outside pinch and
for disabled/fixed fingers; the thumb's own targets are unchanged. Older
profiles default to weight zero. `Finger bend diag` compares the live human
and model proximal-to-distal angles for all four fingers. These are model
diagnostics, not independent measurements of powered physical finger angles.

This adapts the relative-position/orientation and complementary-priority
principles of [Xin et al.](https://arxiv.org/html/2506.09384v2), rather than
reproducing its full arm-hand optimizer. The existing 3-DoF arm controller is
independent. Normalized distal directions, Huber losses, Hermite distance
mapping and local thresholds remain ORCA-specific choices, not paper defaults.
The fingertip origins are kinematic proxies: model coincidence does not prove
physical pad contact. A motor encoder is not an independent link-angle sensor.
No new motor calibration, joint zero, `ref`, mapping or physical contact pose
is inferred from the manual example. Existing 24-pose records are preserved.

```bash
./scripts/teleop_am200_hw.sh --no-arm --pinch-calibrated --hand-feedback-debug
```

Reconnect teleoperation after updating; an already running worker retains its
loaded model/profile. Startup must report `Pinch objective: relative_vector`.
`Hand command trace` reports `final_model_tip_gap_mm`, `target_tip_gap_mm` and
`final_tip_vector_error_mm`; it does not label them measured skin clearance.
The previous surface implementation remains available for explicit comparison:

```bash
./scripts/teleop_am200_hw.sh --no-arm \
  --retarget-config "$PWD/configs/adaptive_analytical_orca_v1_pinch_surface.yaml"
```

For the **preceding surface-distance candidate**, the existing 24 records can
calibrate an empirical distal-skin clearance target:

```bash
# Offline; reuses the existing recording and preserves motor calibration.
./scripts/hand.sh contacts fit-surface
./scripts/teleop_am200_hw.sh --no-arm \
  --retarget-config "$PWD/configs/adaptive_analytical_orca_v1_pinch_surface.yaml" --hand-feedback-debug
# Compare the preceding continuous point objective:
./scripts/teleop_am200_hw.sh --no-arm --pinch-refined --hand-feedback-debug
```

This historical candidate uses
`configs/adaptive_analytical_orca_v1_pinch_surface.yaml`. The backend remains
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

The saved preceding surface YAML uses per-pair `surface_contact_closure_margin_mm`
of **10 mm** for each pair. This candidate is no longer the collector default.
Scalar values remain valid
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
