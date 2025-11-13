# behavior_scoring.py  (fixed)
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
from collections import deque

# ---------- SETTINGS ----------
CAM_INDEX = 0
FPS_TARGET = 15                # target FPS for buffer timing (approx)
WINDOW_SECS = 5                # rolling window length
AGG_INTERVAL = 1.0             # aggregate every N seconds
SAVE_SECONDS = 3               # on flag, save last SAVE_SECONDS of video
OUTPUT_DIR = "flags"           # where to save flagged clips

# thresholds & scoring weights
YAW_THRESH_DEG = 15.0
PITCH_THRESH_DEG = 15.0
MAX_ALLOWED_AWAY = 4.0         # seconds of continuous looking away before heavy penalty
FLAG_BEHAVIOR_THRESH = 0.45    # S_behavior below this triggers flag
W_G, W_A, W_H = 0.5, 0.3, 0.2  # combine weights for S_behavior

# PnP model points and landmark ids (same as previous script)
LANDMARK_IDS = {"nose_tip":1,"chin":152,"left_eye_outer":33,"right_eye_outer":263,"left_mouth":61,"right_mouth":291}
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

# ---------- HELPERS ----------
def rotation_vector_to_euler(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees(x), np.degrees(y), np.degrees(z)

def get_image_points(landmarks, w, h):
    pts = []
    for name in ["nose_tip","chin","left_eye_outer","right_eye_outer","left_mouth","right_mouth"]:
        idx = LANDMARK_IDS[name]
        lm = landmarks[idx]
        pts.append((lm.x * w, lm.y * h))
    return np.array(pts, dtype=np.float64)

def compute_hand_motion_energy(prev_hands, cur_hands):
    if not prev_hands or not cur_hands:
        return 0.0
    energies = []
    for cur in cur_hands:
        cur_pts = np.array([[p.x,p.y,p.z] for p in cur.landmark])
        best_d = None
        for prev in prev_hands:
            prev_pts = np.array([[p.x,p.y,p.z] for p in prev.landmark])
            d = np.linalg.norm((cur_pts - prev_pts).reshape(-1,3), axis=1).mean()
            if best_d is None or d < best_d:
                best_d = d
        if best_d is not None:
            energies.append(best_d)
    return float(np.mean(energies)) if energies else 0.0

# scoring functions (normalized to [0,1])
def gaze_score(fraction_looking, t_min=0.6):
    return max(0.0, min(1.0, (fraction_looking - t_min) / (1.0 - t_min)))

def away_score(max_contiguous_away, max_allowed=MAX_ALLOWED_AWAY):
    penalty = max(0.0, min(1.0, max_contiguous_away / max_allowed))
    return 1.0 - penalty

def hand_score(hand_motion_norm, hand_presence_frac, w_motion=0.7, w_presence=0.3):
    val = 1.0 - (w_motion * hand_motion_norm + w_presence * hand_presence_frac)
    return max(0.0, min(1.0, val))

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ---------- MAIN ----------
def main():
    ensure_dir(OUTPUT_DIR)
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {CAM_INDEX}")
        return

    mp_face = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # circular buffer: store dictionaries with 't', 'frame', 'gaze', 'hand_present', 'hand_motion'
    buffer_len = int(WINDOW_SECS * FPS_TARGET) + 4
    buf = deque(maxlen=buffer_len)

    prev_hand_landmarks = []
    last_agg = time.time()
    last_save_ts = 0
    frame_idx = 0
    fps = 0.0
    prev_time = time.time()

    print("Running behavior scoring. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: frame not read")
            break
        frame_idx += 1
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detection
        fresults = face_mesh.process(img_rgb)
        hresults = hands.process(img_rgb)

        # default per-frame features
        gaze_on = False
        yaw_deg = pitch_deg = roll_deg = 0.0
        face_diag = max(w,h)

        # face pose -> gaze proxy
        if fresults.multi_face_landmarks:
            landmarks = fresults.multi_face_landmarks[0].landmark
            try:
                img_pts = get_image_points(landmarks, w, h)
                focal = w
                camera_matrix = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float64)
                dist_coeffs = np.zeros((4,1))
                ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    pitch_deg, yaw_deg, roll_deg = rotation_vector_to_euler(rvec)
                    gaze_on = (abs(yaw_deg) <= YAW_THRESH_DEG) and (abs(pitch_deg) <= PITCH_THRESH_DEG)
                xs = [lm.x for lm in landmarks]; ys = [lm.y for lm in landmarks]
                face_diag = math.hypot((max(xs)-min(xs))*w, (max(ys)-min(ys))*h)
            except Exception:
                pass
            mp_drawing.draw_landmarks(frame, fresults.multi_face_landmarks[0], mp_face.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))

        # hands
        hand_count = 0
        cur_hands = []
        if hresults.multi_hand_landmarks:
            hand_count = len(hresults.multi_hand_landmarks)
            for hlm in hresults.multi_hand_landmarks:
                cur_hands.append(hlm)
                mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

        raw_motion = compute_hand_motion_energy(prev_hand_landmarks, cur_hands)
        # convert normalized landmark displacement to approx pixel displacement using face_diag
        hand_motion_pix = raw_motion * face_diag
        # a small normalization to keep values roughly in [0, ~1]
        hand_motion_norm = hand_motion_pix / (face_diag + 1e-6)

        # append to buffer (store BGR frame copy)
        ts = time.time()
        buf.append({
            "t": ts,
            "frame": frame.copy(),
            "gaze": bool(gaze_on),
            "hand_present": bool(hand_count > 0),
            "hand_motion": float(hand_motion_norm)
        })

        prev_hand_landmarks = cur_hands

        # initialize default aggregate and score values (safe defaults)
        S_gaze = 1.0
        S_away = 1.0
        S_hand = 1.0
        S_behavior = 1.0
        fraction_looking = 1.0
        max_contig_secs = 0.0
        hand_presence_frac = 0.0
        hand_motion_avg = 0.0

        # aggregate every AGG_INTERVAL seconds
        now = time.time()
        if now - last_agg >= AGG_INTERVAL:
            last_agg = now
            # compute aggregates from buffer
            frames = list(buf)
            N = len(frames)
            if N > 0:
                fraction_looking = sum(1 for x in frames if x["gaze"]) / N
                # compute max contiguous away (in frames -> seconds)
                max_contig = 0
                cur = 0
                for x in frames:
                    if not x["gaze"]:
                        cur += 1
                    else:
                        if cur > max_contig: max_contig = cur
                        cur = 0
                if cur > max_contig: max_contig = cur
                max_contig_secs = max_contig / max(1, FPS_TARGET)

                hand_presence_frac = sum(1 for x in frames if x["hand_present"]) / N
                hand_motion_avg = sum(x["hand_motion"] for x in frames) / N

                # scores
                S_gaze = gaze_score(fraction_looking)
                S_away = away_score(max_contig_secs)
                S_hand = hand_score(hand_motion_avg, hand_presence_frac)
                S_behavior = W_G*S_gaze + W_A*S_away + W_H*S_hand

                # overlay text (will also print)
                reason = []
                if S_behavior < FLAG_BEHAVIOR_THRESH:
                    reason.append("LOW_BEHAVIOR_SCORE")
                if max_contig_secs > MAX_ALLOWED_AWAY:
                    reason.append("LONG_AWAY")
                if hand_presence_frac > 0.8 and hand_motion_avg > 0.02:
                    reason.append("MANY_HANDS_STRONG_MOTION")

                # print summary
                print(f"[agg] t={time.strftime('%H:%M:%S')}, frac_look={fraction_looking:.2f}, max_away={max_contig_secs:.2f}s, hand_frac={hand_presence_frac:.2f}, hand_motion={hand_motion_avg:.4f} => S_beh={S_behavior:.3f} reasons={reason}")

                # flagging logic: save last SAVE_SECONDS seconds if flagged
                if (S_behavior < FLAG_BEHAVIOR_THRESH or max_contig_secs > MAX_ALLOWED_AWAY) and (now - last_save_ts > SAVE_SECONDS):
                    last_save_ts = now
                    save_clip(frames, OUTPUT_DIR, SAVE_SECONDS, FPS_TARGET)
            else:
                # buffer empty - keep defaults
                pass

        # compute FPS
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # overlay info on frame
        y = 30
        def put(s, color=(0,255,0)):
            nonlocal y
            cv2.putText(frame, s, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 22

        put(f"FPS:{fps:.1f}")
        put(f"Yaw:{yaw_deg:.1f} Pitch:{pitch_deg:.1f}")
        # latest aggregates quick view
        if buf:
            last = buf[-1]
            put(f"Gaze:{last['gaze']} Hands:{last['hand_present']} HandMotion:{last['hand_motion']:.4f}")
        put(f"S_beh (live agg every {AGG_INTERVAL}s): {S_behavior:.3f}", color=(0,200,255) if S_behavior>0.5 else (0,0,255))

        # circle indicator
        color = (0,255,0) if (buf and buf[-1]["gaze"]) else (0,0,255)
        cv2.circle(frame, (w-40,40), 16, color, -1)

        cv2.imshow("Behavior Scoring - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    face_mesh.close(); hands.close()
    cap.release(); cv2.destroyAllWindows()
    print("Exiting.")

# helper to save last few seconds as mp4
def save_clip(frames, out_dir, seconds, fps):
    if not frames:
        return
    ensure_dir(out_dir)
    # take last seconds worth of frames (approx)
    count = max(1, int(seconds * fps))
    clip_frames = frames[-count:]
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(out_dir, f"flag_{ts}.mp4")
    h, w = clip_frames[0]["frame"].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))
    for f in clip_frames:
        writer.write(f["frame"])
    writer.release()
    print(f"[flag] saved clip {out_path}")

if __name__ == "__main__":
    main()
