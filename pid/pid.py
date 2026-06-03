"""
LAB: Approach & land in front of an AprilTag using PID + pose
=============================================================
Your drone starts at a RANDOM position and heading. Make it:

  1. Take off          (press 1)
  2. Start auto-mode   (press SPACE) -> it spins until it sees AprilTag 14,
                                        then approaches and lands in front.

You write THREE things:
  * STUDENT TODO 1 -> TELLO_CAMERA_PARAMS (your own camera calibration)
  * STUDENT TODO 2 -> PID.update()        (the PID controller)
  * STUDENT TODO 3 -> compute_control()   (turn the detection into a command)

Everything else (tag detection, manual control, main loop, HUD) is provided.

Manual keys (focus the OpenCV window) -- keep these handy for safety:
  1 takeoff   2 land   3 stop   SPACE toggle auto-mode
  w/s fwd/back  a/d strafe left/right  z/x down/up  c/v yaw CW/CCW
  e EMERGENCY (motors off -- the drone WILL fall!)   q quit (lands)

Fly in a clear, open area. Install:
  pip install djitellopy numpy opencv-python pupil-apriltags
"""

import logging
import time

import cv2
import numpy as np
from djitellopy import Tello
from pupil_apriltags import Detector

# ============================================================
#  CONFIGURATION  (tune the gains during the lab)
# ============================================================
FRAME_W, FRAME_H = 960, 720

TARGET_TAG_ID = 14
APRILTAG_FAMILY = "tag36h11"

# ===== STUDENT TODO 1: camera calibration ==========================
# Replace None with your tuple, e.g. (956.1, 921.1, 471.5, 380.3).
TELLO_CAMERA_PARAMS = None  # <-- TODO 1: (fx, fy, cx, cy)
# ===== END STUDENT TODO 1 ==========================================
APRILTAG_PHYSICAL_SIZE = 0.155  # tag side length in METERS (measure yours!)

APRILTAG_TARGET_DISTANCE = 0.5  # stop this far from the tag (meters)
LAND_DISTANCE_MARGIN = 0.13  # land once tz <= TARGET + this

YAW_GAINS = (0.25, 0.0, 0.08)  # (kp, ki, kd) for horizontal centering -> yaw
UD_GAINS = (0.35, 0.0, 0.10)  # (kp, ki, kd) for vertical centering   -> up/down

APRILTAG_FB_GAIN = 20.0  # distance error (m) -> forward RC units
APRILTAG_LR_GAIN = -20.0  # tag-normal lateral -> strafe RC (FLIP SIGN if it runs away)
APRILTAG_LR_DEADZONE = 0.05

MAX_CMD = 40  # cap on lr / fb / ud
YAW_MAX_CMD = 55  # cap on yaw
ERROR_DEADZONE = 18  # ignore pixel errors smaller than this
SEARCH_YAW = 40  # spin speed while looking for the tag
DT_MIN, DT_MAX = 0.01, 0.20


# ============================================================
#  PID CONTROLLER
# ============================================================
class PID:
    def __init__(self, kp, ki, kd, output_limit=MAX_CMD, integral_limit=40.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._initialised = False

    def update(self, error, dt):
        # ===== STUDENT TODO 2: implement the PID update =====================
        # Available: self.kp/ki/kd, self._integral, self._prev_error,
        #            self._initialised, self.integral_limit, self.output_limit
        #
        #   1. p = kp * error
        #   2. self._integral += error * dt
        #      clamp self._integral to +/- self.integral_limit  (anti-windup)
        #      i = ki * self._integral
        #   3. if self._initialised and dt > 0:
        #          d = kd * (error - self._prev_error) / dt
        #      else:
        #          d = 0.0
        #   4. self._prev_error = error ; self._initialised = True
        #   5. return p + i + d, clamped to +/- self.output_limit
        #
        # Tip: np.clip(value, low, high)
        raise NotImplementedError("STUDENT TODO 2: implement PID.update")
        # ====================================================================


# ============================================================
#  APRILTAG DETECTOR (provided)
# ============================================================
class AprilTagDetector:
    def __init__(self, family, camera_params, tag_size):
        if camera_params is None:
            raise NotImplementedError(
                "STUDENT TODO 1: set TELLO_CAMERA_PARAMS to your calibrated "
                "(fx, fy, cx, cy) before running."
            )

        print(f"Loading AprilTag detector: family={family}")
        self.detector = Detector(families=family)
        self.camera_params = camera_params
        self.tag_size = tag_size

    def __call__(self, bgr_frame, target_id=None):
        """Return (cx, cy, w, h, tag_id, pose_R, pose_t) for the largest
        matching tag, else None.

        pose_t : tag position in the camera frame, METERS. pose_t[2] (tz)
                 is the distance straight ahead -- bigger = farther.
        pose_R : tag orientation. pose_R[0, 2] is the x-part of the tag's
                 outward normal; it's ~0 when you face the tag head-on,
                 and its sign tells you which way to strafe.
        """
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        results = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size,
        )
        if target_id is not None:
            results = [r for r in results if r.tag_id == target_id]
        if not results:
            return None

        def _bbox(r):
            cs = np.asarray(r.corners)
            xs, ys = cs[:, 0], cs[:, 1]
            return (
                float((xs.min() + xs.max()) / 2.0),
                float((ys.min() + ys.max()) / 2.0),
                float(xs.max() - xs.min()),
                float(ys.max() - ys.min()),
            )

        best = max(results, key=lambda r: (lambda b: b[2] * b[3])(_bbox(r)))
        cx, cy, w, h = _bbox(best)
        return (cx, cy, w, h, int(best.tag_id), best.pose_R, best.pose_t)


# ============================================================
#  MANUAL KEYBOARD CONTROL (provided -- keep for safety)
# ============================================================
def manual_keyboard(drone: Tello, key: int) -> bool:
    """Handle a manual key. Returns True if it was a manual command, so the
    auto controller doesn't also send one on the same frame."""
    fb, lr, ud, yaw = 30, 40, 50, 30
    if key == ord("1"):
        drone.takeoff()
        return True
    if key == ord("2"):
        drone.land()
        return True
    if key == ord("3"):
        drone.send_rc_control(0, 0, 0, 0)
        return True
    if key == ord("w"):
        drone.send_rc_control(0, fb, 0, 0)
        return True
    if key == ord("s"):
        drone.send_rc_control(0, -fb, 0, 0)
        return True
    if key == ord("a"):
        drone.send_rc_control(-lr, 0, 0, 0)
        return True
    if key == ord("d"):
        drone.send_rc_control(lr, 0, 0, 0)
        return True
    if key == ord("z"):
        drone.send_rc_control(0, 0, -ud, 0)
        return True
    if key == ord("x"):
        drone.send_rc_control(0, 0, ud, 0)
        return True
    if key == ord("c"):
        drone.send_rc_control(0, 0, 0, yaw)
        return True
    if key == ord("v"):
        drone.send_rc_control(0, 0, 0, -yaw)
        return True
    if key == ord("e"):
        print("EMERGENCY!")
        drone.emergency()
        return True
    return False


# ============================================================
#  CONTROLLER
# ============================================================
class AprilTagApproach:
    def __init__(self):
        self.pid_yaw = PID(*YAW_GAINS, output_limit=YAW_MAX_CMD)
        self.pid_ud = PID(*UD_GAINS)
        self.detector = AprilTagDetector(
            APRILTAG_FAMILY, TELLO_CAMERA_PARAMS, APRILTAG_PHYSICAL_SIZE
        )
        self.tello = None
        self.running = False
        self.tracking_enabled = False  # toggled with SPACE

    @staticmethod
    def _deadzone(v, t):
        return 0.0 if abs(v) < t else v

    def compute_control(self, detection, dt):
        """STUDENT TODO 3 -- the control law.

        `detection` is None (no tag) or (cx, cy, w, h, tag_id, pose_R, pose_t).
        Return an (lr, fb, ud, yaw) command as numbers -- the framework
        clamps and sends them.

        A) detection is None  -> spin to SEARCH:  return (0, 0, 0, SEARCH_YAW)

        B) tag visible -> center it, then approach using the pose:
             cx, cy = detection[0], detection[1]
             pose_R, pose_t = detection[5], detection[6]

             err_x = self._deadzone(cx - FRAME_W/2, ERROR_DEADZONE)
             err_y = self._deadzone(cy - FRAME_H/2, ERROR_DEADZONE)
             yaw = self.pid_yaw.update(err_x, dt)
             ud  = -self.pid_ud.update(err_y, dt)    # image y grows downward

             tz = float(np.asarray(pose_t).flatten()[2])   # metres ahead
             if tz <= APRILTAG_TARGET_DISTANCE + LAND_DISTANCE_MARGIN:
                 self.running = False                # close enough -> land
                 return (0, 0, 0, 0)
             fb = APRILTAG_FB_GAIN * (tz - APRILTAG_TARGET_DISTANCE)

             lateral = float(pose_R[0, 2])            # ~0 when square-on
             if abs(lateral) < APRILTAG_LR_DEADZONE:
                 lateral = 0.0
             lr = APRILTAG_LR_GAIN * lateral

             return (lr, fb, ud, yaw)
        """
        raise NotImplementedError("STUDENT TODO 3: implement compute_control")

    # ----- main loop (provided) -------------------------------
    def _send(self, lr, fb, ud, yaw):
        c = lambda v, lim: int(np.clip(v, -lim, lim))
        self.tello.send_rc_control(
            c(lr, MAX_CMD), c(fb, MAX_CMD), c(ud, MAX_CMD), c(yaw, YAW_MAX_CMD)
        )

    def run(self):
        self.tello = Tello()
        self.tello.LOGGER.setLevel(logging.CRITICAL)
        self.tello.connect()
        print(f"Battery: {self.tello.get_battery()}%")
        self.tello.streamon()
        frame_reader = self.tello.get_frame_read()
        time.sleep(2.0)

        print("Ready. 1 = takeoff, SPACE = auto-mode, q = quit.")
        self.running = True
        last_t = time.time()

        try:
            while self.running:
                frame_rgb = frame_reader.frame
                if frame_rgb is None:
                    continue
                frame = cv2.resize(
                    cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), (FRAME_W, FRAME_H)
                )

                now = time.time()
                dt = float(np.clip(now - last_t, DT_MIN, DT_MAX))
                last_t = now

                detection = self.detector(frame, target_id=TARGET_TAG_ID)

                if self.tracking_enabled:
                    lr, fb, ud, yaw = self.compute_control(detection, dt)
                else:
                    lr = fb = ud = yaw = 0

                self._draw_debug(frame, detection, (lr, fb, ud, yaw))
                cv2.imshow("AprilTag Lab", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    self.tracking_enabled = not self.tracking_enabled
                    self.pid_yaw.reset()
                    self.pid_ud.reset()
                    print("Auto-mode", "ON" if self.tracking_enabled else "OFF")
                    self.tello.send_rc_control(0, 0, 0, 0)
                    continue
                if manual_keyboard(self.tello, key):
                    continue  # a manual command was sent this frame
                if self.tracking_enabled:
                    self._send(lr, fb, ud, yaw)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def _draw_debug(self, frame, detection, cmd):
        cv2.drawMarker(
            frame,
            (FRAME_W // 2, FRAME_H // 2),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            20,
            1,
        )
        if detection is not None:
            cx, cy, w, h = detection[:4]
            cv2.rectangle(
                frame,
                (int(cx - w / 2), int(cy - h / 2)),
                (int(cx + w / 2), int(cy + h / 2)),
                (0, 255, 0),
                2,
            )
            if detection[6] is not None and detection[5] is not None:
                tz = float(np.asarray(detection[6]).flatten()[2])
                lateral = float(detection[5][0, 2])
                cv2.putText(
                    frame,
                    f"tz={tz:.2f}m lateral={lateral:+.2f} target={APRILTAG_TARGET_DISTANCE}m",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
        lr, fb, ud, yaw = cmd
        mode = "AUTO" if self.tracking_enabled else "MANUAL"
        cv2.putText(
            frame,
            f"[{mode}] lr={int(lr)} fb={int(fb)} ud={int(ud)} yaw={int(yaw)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    def shutdown(self):
        self.running = False
        if self.tello is not None:
            try:
                self.tello.send_rc_control(0, 0, 0, 0)
                self.tello.land()
                self.tello.streamoff()
            except Exception as e:
                print(f"Shutdown warning: {e}")
        cv2.destroyAllWindows()
        print("Landed and stopped.")


if __name__ == "__main__":
    AprilTagApproach().run()
