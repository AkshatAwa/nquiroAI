# behaviour_scoring_calib_full.py
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
from collections import deque

# ------------------ IRIS MODEL BACKUP (ONNX / PyTorch) ------------------
# This block embeds a lightweight IrisPredictor so no extra files are required.
ONNX_AVAILABLE = False
TORCH_AVAILABLE = False
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except Exception:
    try:
        import torch
        TORCH_AVAILABLE = True
    except Exception:
        pass

# Try to import torchvision.transforms only if torch is available
_TORCHVISION_AVAILABLE = False
if TORCH_AVAILABLE:
    try:
        import torchvision.transforms as T
        _TORCHVISION_AVAILABLE = True
    except Exception:
        _TORCHVISION_AVAILABLE = False

class IrisPredictor:
    """
    Minimal embedded iris predictor:
    - tries ONNX runtime first if iris_model.onnx exists
    - falls back to PyTorch if iris_model.pth exists and torch available
    - predict(eye_crop_bgr) -> list of (x_px, y_px) coordinates in crop
    NOTE: this is a compatibility/fallback wrapper. If you have a custom architecture,
    adapt the loading code inside this class to match.
    """
    def __init__(self, onnx_path="iris_model.onnx", pth_path="iris_model.pth", device='cpu'):
        self.backend = None
        self.session = None
        self.model = None
        self.device = device
        self.input_size = (64, 64)  # change if your model expects different
        self.transform = None

        if ONNX_AVAILABLE and os.path.exists(onnx_path):
            try:
                self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                self.backend = 'onnx'
                print("[IrisPredictor] Loaded ONNX model:", onnx_path)
            except Exception as e:
                print("[IrisPredictor] Failed to load ONNX:", e)
                self.session = None

        if self.session is None and TORCH_AVAILABLE and os.path.exists(pth_path):
            try:
                import torch
                state = torch.load(pth_path, map_location=torch.device(device))
                # If the file is a state_dict, user must adapt model architecture.
                # We'll try to use a tiny fallback head if possible, else store object.
                if isinstance(state, dict):
                    # attempt to detect saved model object vs state_dict
                    # If state contains recognizable keys, attempt to load to a tiny net
                    class TinyNet(torch.nn.Module):
                        def __init__(self, out_dim=4):
                            super().__init__()
                            self.net = torch.nn.Sequential(
                                torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.ReLU(),
                                torch.nn.AdaptiveAvgPool2d(1),
                                torch.nn.Flatten(),
                                torch.nn.Linear(16, out_dim)
                            )
                        def forward(self, x): return self.net(x)

                    # default assume 4 outputs (2 keypoints). If your model uses more, change accordingly.
                    out_dim = 4
                    try:
                        m = TinyNet(out_dim).to(device)
                        m.load_state_dict(state)
                        self.model = m
                        self.backend = 'torch'
                        print("[IrisPredictor] Loaded PyTorch state_dict into TinyNet fallback.")
                    except Exception:
                        # maybe state contains 'model' key
                        if 'model' in state and isinstance(state['model'], dict):
                            try:
                                m.load_state_dict(state['model'])
                                self.model = m
                                self.backend = 'torch'
                                print("[IrisPredictor] Loaded nested state['model'] into TinyNet fallback.")
                            except Exception:
                                # fallback: store raw state
                                self.model = state
                                self.backend = 'torch'
                                print("[IrisPredictor] Loaded PyTorch state (unwrapped).")
                        else:
                            # fallback: store raw state
                            self.model = state
                            self.backend = 'torch'
                            print("[IrisPredictor] Loaded PyTorch state (raw).")
                else:
                    # maybe file contains full model object (torchscript or pickled)
                    self.model = state
                    self.backend = 'torch'
                    print("[IrisPredictor] Loaded PyTorch model object from file.")
                # prepare transform if torchvision available
                if _TORCHVISION_AVAILABLE:
                    self.transform = T.Compose([T.ToPILImage(), T.Resize(self.input_size), T.ToTensor()])
            except Exception as e:
                print("[IrisPredictor] Failed to load PyTorch model:", e)
                self.model = None

        if self.backend is None:
            print("[IrisPredictor] No iris model available (ONNX/PyTorch not found). Using MediaPipe-only fallback.")
        else:
            print(f"[IrisPredictor] Backend set to: {self.backend}")

    def predict(self, eye_crop_bgr):
        """
        Return list of (x_px, y_px) coords in the crop coordinate system.
        On failure or missing model, returns [].
        """
        if self.backend is None:
            return []
        try:
            H, W = eye_crop_bgr.shape[:2]
            if self.backend == 'onnx':
                img = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.input_size)
                img = img.astype(np.float32) / 255.0
                x = np.transpose(img, (2,0,1)).astype(np.float32)[None, ...]  # NCHW
                input_name = self.session.get_inputs()[0].name
                outs = self.session.run(None, {input_name: x})
                out = np.array(outs[0]).reshape(-1)
            elif self.backend == 'torch':
                import torch
                if self.transform is None:
                    # minimal preprocess without torchvision
                    img = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.input_size).astype(np.float32)/255.0
                    x = np.transpose(img, (2,0,1))[None, ...]
                    xt = torch.from_numpy(x).float().to(self.device)
                else:
                    xt = self.transform(cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
                self.model.eval()
                with torch.no_grad():
                    out_t = self.model(xt)
                out = out_t.cpu().numpy().reshape(-1)
            else:
                return []
            coords = []
            for i in range(0, len(out), 2):
                nx, ny = float(out[i]), float(out[i+1])
                # handle normalized [-1,1] or [0,1]
                if nx < 0 or nx > 1:
                    nx = (nx + 1.0) / 2.0
                if ny < 0 or ny > 1:
                    ny = (ny + 1.0) / 2.0
                px = max(0.0, min(1.0, nx)) * W
                py = max(0.0, min(1.0, ny)) * H
                coords.append((px, py))
            return coords
        except Exception as e:
            # don't spam errors — return empty to fall back to pose
            # print("[IrisPredictor] predict error:", e)
            return []

# ------------------ CONFIG / SETTINGS ------------------
# ---------- SETTINGS ----------
CAM_INDEX = 0
FPS_TARGET = 15
WINDOW_SECS = 5
AGG_INTERVAL = 1.0
SAVE_SECONDS = 3
OUTPUT_DIR = "flags"

# Default thresholds (calibration will set neutral offsets)
DEFAULT_YAW_THRESH = 25.0
DEFAULT_PITCH_THRESH = 25.0
MAX_ALLOWED_AWAY = 4.0
FLAG_BEHAVIOR_THRESH = 0.45
W_G, W_A, W_H = 0.5, 0.3, 0.2

# PnP constants (same as before)
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
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

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
    return np.degrees(x), np.degrees(y), np.degrees(z)  # pitch, yaw, roll

def get_image_points(landmarks, w, h):
    pts = []
    for name in ["nose_tip","chin","left_eye_outer","right_eye_outer","left_mouth","right_mouth"]:
        idx = LANDMARK_IDS[name]
        lm = landmarks[idx]
        pts.append((lm.x * w, lm.y * h))
    return np.array(pts, dtype=np.float64)

# scoring funcs
def gaze_score(fraction_looking, t_min=0.6):
    return max(0.0, min(1.0, (fraction_looking - t_min) / (1.0 - t_min)))
def away_score(max_contiguous_away, max_allowed=MAX_ALLOWED_AWAY):
    penalty = max(0.0, min(1.0, max_contiguous_away / max_allowed))
    return 1.0 - penalty
def hand_score(hand_motion_norm, hand_presence_frac, w_motion=0.7, w_presence=0.3):
    v = 1.0 - (w_motion * hand_motion_norm + w_presence * hand_presence_frac)
    return max(0.0, min(1.0, v))

# save clip helper
def save_clip(frames, out_dir, seconds, fps):
    if not frames:
        return
    ensure_dir(out_dir)
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

# ---------- CALIBRATION ----------
def calibrate_neutral(cap, face_mesh, seconds=4):
    print(f"Calibration: Please sit naturally and look at the camera for {seconds} seconds.")
    start = time.time()
    yaws = []
    pitches = []
    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(img_rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            try:
                img_pts = get_image_points(lm, w, h)
                focal = w
                camera_matrix = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float64)
                dist_coeffs = np.zeros((4,1))
                ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    pitch, yaw, roll = rotation_vector_to_euler(rvec)
                    yaws.append(yaw); pitches.append(pitch)
            except Exception:
                pass
        cv2.putText(frame, "Calibration: look at camera", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow("Calibration")
    if yaws and pitches:
        neutral_yaw = float(np.mean(yaws))
        neutral_pitch = float(np.mean(pitches))
        print(f"Calibration done. neutral_yaw={neutral_yaw:.2f}, neutral_pitch={neutral_pitch:.2f} (deg)")
        return neutral_yaw, neutral_pitch
    else:
        print("Calibration failed to get face pose. Using neutral (0,0).")
        return 0.0, 0.0

# ---------- MAIN ----------
def main():
    ensure_dir(OUTPUT_DIR)
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: cannot open camera", CAM_INDEX)
        return

    mp_face = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # prepare iris predictor (ONNX -> PyTorch fallback)
    iris_pred = IrisPredictor()

    # calibrate neutral pose
    neutral_yaw, neutral_pitch = calibrate_neutral(cap, face_mesh, seconds=4)
    yaw_thresh = DEFAULT_YAW_THRESH
    pitch_thresh = DEFAULT_PITCH_THRESH
    print(f"Using thresholds yaw:{yaw_thresh}°, pitch:{pitch_thresh}° around neutral.")

    # buffers & state
    buf = deque(maxlen=int(WINDOW_SECS * FPS_TARGET) + 4)
    prev_hand_landmarks = []
    last_agg = time.time()
    last_save_ts = 0
    fps = 0.0
    prev_time = time.time()

    # iris indices and eye corners (MediaPipe refine landmarks)
    LEFT_IRIS_IDX = [468, 469, 470, 471]
    RIGHT_IRIS_IDX = [473, 474, 475, 476]
    LEFT_EYE_CORNERS = (33, 133)
    RIGHT_EYE_CORNERS = (362, 263)

    print("Starting detection. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        fresults = face_mesh.process(img_rgb)
        hresults = hands.process(img_rgb)

        gaze_on = False
        yaw_deg = pitch_deg = roll_deg = 0.0
        face_diag = max(w,h)
        used_iris = False
        used_pose = False

        if fresults.multi_face_landmarks:
            lm = fresults.multi_face_landmarks[0].landmark
            try:
                # ---------- Try iris-based gaze using MediaPipe first ----------
                def iris_center(indices):
                    pts = []
                    for i in indices:
                        if i < len(lm):
                            pts.append((lm[i].x * w, lm[i].y * h))
                    if not pts:
                        return None
                    return np.array(pts).mean(axis=0)

                left_iris = iris_center(LEFT_IRIS_IDX)
                right_iris = iris_center(RIGHT_IRIS_IDX)

                def eye_box(corner_inds):
                    a = lm[corner_inds[0]]
                    b = lm[corner_inds[1]]
                    x_min = min(a.x, b.x) * w
                    x_max = max(a.x, b.x) * w
                    y_min = min(a.y, b.y) * h
                    y_max = max(a.y, b.y) * h
                    return (x_min, x_max, y_min, y_max)

                gaze_on_iris = False
                iris_valid_count = 0
                H_CENTRAL_RATIO = 0.45
                V_CENTRAL_RATIO = 0.55

                if left_iris is not None:
                    x_min, x_max, y_min, y_max = eye_box(LEFT_EYE_CORNERS)
                    if x_max - x_min > 1 and y_max - y_min > 1:
                        cx = (left_iris[0] - (x_min + x_max)/2) / (x_max - x_min)
                        cy = (left_iris[1] - (y_min + y_max)/2) / (y_max - y_min)
                        if abs(cx) <= H_CENTRAL_RATIO/2 and abs(cy) <= V_CENTRAL_RATIO/2:
                            gaze_on_iris = True
                        iris_valid_count += 1

                if right_iris is not None:
                    x_min, x_max, y_min, y_max = eye_box(RIGHT_EYE_CORNERS)
                    if x_max - x_min > 1 and y_max - y_min > 1:
                        cx = (right_iris[0] - (x_min + x_max)/2) / (x_max - x_min)
                        cy = (right_iris[1] - (y_min + y_max)/2) / (y_max - y_min)
                        if abs(cx) <= H_CENTRAL_RATIO/2 and abs(cy) <= V_CENTRAL_RATIO/2:
                            gaze_on_iris = gaze_on_iris or True
                        iris_valid_count += 1

                if iris_valid_count > 0:
                    # MediaPipe iris landmarks available and used
                    gaze_on = bool(gaze_on_iris)
                    used_iris = True
                else:
                    # ---------- Fallback: try model-based iris predictor if available ----------
                    def make_eye_crop(corner_inds, pad=0.15):
                        a = lm[corner_inds[0]]; b = lm[corner_inds[1]]
                        xmin = int(min(a.x,b.x)*w - pad*w); xmax = int(max(a.x,b.x)*w + pad*w)
                        ymin = int(min(a.y,b.y)*h - pad*h); ymax = int(max(a.y,b.y)*h + pad*h)
                        xmin, ymin = max(0, xmin), max(0, ymin)
                        xmax, ymax = min(w-1, xmax), min(h-1, ymax)
                        if xmax - xmin < 4 or ymax - ymin < 4:
                            return None, (0,0)
                        crop = frame[ymin:ymax, xmin:xmax].copy()
                        return crop, (xmin, ymin)

                    vote = False; valid = 0
                    left_crop, left_off = make_eye_crop(LEFT_EYE_CORNERS)
                    right_crop, right_off = make_eye_crop(RIGHT_EYE_CORNERS)

                    if left_crop is not None and iris_pred.backend is not None:
                        coords = iris_pred.predict(left_crop)
                        if coords:
                            valid += 1
                            cx, cy = coords[0]
                            Wc, Hc = left_crop.shape[1], left_crop.shape[0]
                            rx = (cx - Wc/2) / Wc
                            ry = (cy - Hc/2) / Hc
                            if abs(rx) <= H_CENTRAL_RATIO/2 and abs(ry) <= V_CENTRAL_RATIO/2:
                                vote = True
                            # draw indicator
                            fx, fy = int(left_off[0] + cx), int(left_off[1] + cy)
                            cv2.circle(frame, (fx, fy), 3, (0,255,0) if vote else (0,0,255), -1)

                    if right_crop is not None and iris_pred.backend is not None:
                        coords = iris_pred.predict(right_crop)
                        if coords:
                            valid += 1
                            cx, cy = coords[0]
                            Wc, Hc = right_crop.shape[1], right_crop.shape[0]
                            rx = (cx - Wc/2) / Wc
                            ry = (cy - Hc/2) / Hc
                            if abs(rx) <= H_CENTRAL_RATIO/2 and abs(ry) <= V_CENTRAL_RATIO/2:
                                vote = vote or True
                            fx, fy = int(right_off[0] + cx), int(right_off[1] + cy)
                            cv2.circle(frame, (fx, fy), 3, (0,255,0) if vote else (0,0,255), -1)

                    if valid > 0:
                        gaze_on = bool(vote)
                        used_iris = True
                    else:
                        # Fallback: head pose
                        img_pts = get_image_points(lm, w, h)
                        focal = w
                        camera_matrix = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float64)
                        dist_coeffs = np.zeros((4,1))
                        ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                        if ok:
                            pitch_deg, yaw_deg, roll_deg = rotation_vector_to_euler(rvec)
                            gaze_on = (abs(yaw_deg - neutral_yaw) <= yaw_thresh) and (abs(pitch_deg - neutral_pitch) <= pitch_thresh)
                            used_pose = True

                xs = [p.x for p in lm]; ys = [p.y for p in lm]
                face_diag = math.hypot((max(xs)-min(xs))*w, (max(ys)-min(ys))*h)
            except Exception:
                pass

            mp_drawing.draw_landmarks(frame, fresults.multi_face_landmarks[0], mp_face.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))

            method = "iris" if used_iris else ("pose" if used_pose else "none")
            cv2.putText(frame, f"gaze_method:{method}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

        # Hands -> count & simple motion energy
        hand_count = 0
        cur_hands = []
        if hresults.multi_hand_landmarks:
            hand_count = len(hresults.multi_hand_landmarks)
            for hlm in hresults.multi_hand_landmarks:
                cur_hands.append(hlm)
                mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

        # compute hand motion energy (normalized-ish)
        raw_motion = 0.0
        if prev_hand_landmarks and cur_hands:
            energies = []
            for cur in cur_hands:
                cur_pts = np.array([[p.x,p.y,p.z] for p in cur.landmark])
                best = None
                for prev in prev_hand_landmarks:
                    prev_pts = np.array([[p.x,p.y,p.z] for p in prev.landmark])
                    d = np.linalg.norm((cur_pts - prev_pts).reshape(-1,3), axis=1).mean()
                    if best is None or d < best:
                        best = d
                if best is not None:
                    energies.append(best)
            raw_motion = float(np.mean(energies)) if energies else 0.0
        hand_motion_norm = (raw_motion * face_diag) / (face_diag + 1e-6)

        # append to buffer
        buf.append({
            "t": time.time(),
            "frame": frame.copy(),
            "gaze": bool(gaze_on),
            "hand_present": bool(hand_count > 0),
            "hand_motion": float(hand_motion_norm)
        })
        prev_hand_landmarks = cur_hands

        # aggregation
        now = time.time()
        if now - last_agg >= AGG_INTERVAL:
            last_agg = now
            frames = list(buf)
            N = len(frames)
            if N > 0:
                fraction_looking = sum(1 for x in frames if x["gaze"]) / N
                # max contiguous away
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

                S_gaze = gaze_score(fraction_looking)
                S_away = away_score(max_contig_secs)
                S_hand = hand_score(hand_motion_avg, hand_presence_frac)
                S_behavior = W_G*S_gaze + W_A*S_away + W_H*S_hand

                # DEBUG: print aggregates
                print(f"[agg] t={time.strftime('%H:%M:%S')}, frac_look={fraction_looking:.2f}, max_away={max_contig_secs:.2f}s, hand_frac={hand_presence_frac:.2f}, hand_motion={hand_motion_avg:.4f} => S_beh={S_behavior:.3f}, neutral_y={neutral_yaw:.2f}, neutral_p={neutral_pitch:.2f}")

                if (S_behavior < FLAG_BEHAVIOR_THRESH or max_contig_secs > MAX_ALLOWED_AWAY) and (now - last_save_ts > SAVE_SECONDS):
                    last_save_ts = now
                    save_clip(frames, OUTPUT_DIR, SAVE_SECONDS, FPS_TARGET)

        # FPS
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # overlay
        y = 30
        def put(s, color=(0,255,0)):
            nonlocal y
            cv2.putText(frame, s, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 22
        put(f"FPS:{fps:.1f}")
        put(f"Yaw:{yaw_deg:.1f} Pitch:{pitch_deg:.1f} (neutral_y:{neutral_yaw:.1f} p:{neutral_pitch:.1f})")
        if buf:
            last = buf[-1]
            put(f"Gaze:{last['gaze']} Hands:{last['hand_present']} HM:{last['hand_motion']:.4f}")
        # indicator
        color = (0,255,0) if (buf and buf[-1]["gaze"]) else (0,0,255)
        cv2.circle(frame, (w-40,40), 16, color, -1)

        cv2.imshow("Behavior Calibrated - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    face_mesh.close(); hands.close()
    cap.release(); cv2.destroyAllWindows()
    print("Exited.")

if __name__ == "__main__":
    main()
