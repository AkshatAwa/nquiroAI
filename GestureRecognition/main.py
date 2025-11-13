# behaviour_scoring_combined.py
# Combined pipeline: MediaPipe FaceMesh (primary) + IrisModel fallback (ONNX or PyTorch)
# Save as behaviour_scoring_combined.py and run with python.

import cv2, time, math, os, sys, numpy as np
from collections import deque

# Try imports for model inference
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

# mediapipe
try:
    import mediapipe as mp
except Exception as e:
    print("Please install MediaPipe (pip install mediapipe).", e)
    sys.exit(1)

# ------------------ CONFIG (tune these) ------------------
CAM_INDEX = 0
FPS_TARGET = 15

WINDOW_SECS = 8
AGG_INTERVAL = 2.0
SAVE_SECONDS = 4
OUTPUT_DIR = "flags_combined"

EMA_ALPHA = 0.25
MEDIAN_K = 3

DEFAULT_YAW_THRESH = 30.0
DEFAULT_PITCH_THRESH = 30.0
H_CENTRAL_RATIO = 0.55
V_CENTRAL_RATIO = 0.65

MAX_ALLOWED_AWAY = 6.0
FLAG_BEHAVIOR_THRESH = 0.35
W_G, W_A, W_H = 0.6, 0.25, 0.15

SAVE_COOLDOWN = 10.0

# model filenames (place in same dir as script)
IRIS_ONNX_PATH = "iris_model.onnx"
IRIS_PTH_PATH = "iris_model.pth"  # fallback if ONNX not present

# PnP constants and landmark IDs
LANDMARK_IDS = {"nose_tip":1,"chin":152,"left_eye_outer":33,"right_eye_outer":280,"left_mouth":61,"right_mouth":291}
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

# Eye/iris indices (MediaPipe refine)
LEFT_IRIS_IDX = [468,469,470,471]
RIGHT_IRIS_IDX = [473,474,475,476]
LEFT_EYE_CORNERS=(33,133); RIGHT_EYE_CORNERS=(362,263)

# ---------------------------------------------------------

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

def gaze_score(fraction_looking, t_min=0.6):
    return max(0.0, min(1.0, (fraction_looking - t_min) / (1.0 - t_min)))
def away_score(max_contiguous_away, max_allowed=MAX_ALLOWED_AWAY):
    penalty = max(0.0, min(1.0, max_contiguous_away / max_allowed))
    return 1.0 - penalty
def hand_score(hand_motion_norm, hand_presence_frac, w_motion=0.7, w_presence=0.3):
    v = 1.0 - (w_motion * hand_motion_norm + w_presence * hand_presence_frac)
    return max(0.0, min(1.0, v))

def save_clip(frames, out_dir, seconds, fps):
    if not frames: return
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

# ---------------- Iris predictor wrapper ----------------
class IrisPredictor:
    """
    Tries ONNX runtime first (iris_model.onnx).
    Falls back to PyTorch if available (iris_model.pth).
    The model is expected to output normalized coordinates [0..1] for each keypoint: (x1,y1,x2,y2,...).
    """
    def __init__(self, onnx_path=IRIS_ONNX_PATH, pth_path=IRIS_PTH_PATH, device='cpu'):
        self.backend = None
        self.session = None
        self.model = None
        self.device = device
        if ONNX_AVAILABLE and os.path.exists(onnx_path):
            try:
                self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                self.backend = 'onnx'
                print("Loaded iris ONNX model:", onnx_path)
            except Exception as e:
                print("Failed to load ONNX model:", e)
                self.session = None
        if self.session is None and TORCH_AVAILABLE and os.path.exists(pth_path):
            import torch
            try:
                # Attempt to load state_dict into a simple conv model if possible.
                # If the model class differs, the user should modify this class to match training code.
                # We'll try to load state_dict; if fails, attempt torch.load full object.
                state = torch.load(pth_path, map_location=torch.device(device))
                if isinstance(state, dict):
                    # unknown architecture: create a tiny convolutional head compatible with typical keypoint regressors
                    # This is a fallback and likely requires replacing with your exact architecture.
                    class TinyNet(torch.nn.Module):
                        def __init__(self, out_dim):
                            super().__init__()
                            self.net = torch.nn.Sequential(
                                torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.ReLU(),
                                torch.nn.AdaptiveAvgPool2d(1),
                                torch.nn.Flatten(),
                                torch.nn.Linear(16, out_dim)
                            )
                        def forward(self,x): return self.net(x)
                    # Determine output dim from the saved state (if possible)
                    out_dim = None
                    # try to infer
                    for k in state.keys():
                        if 'net.' in k and 'weight' in k:
                            # heuristic fallback: set out_dim = length of matching bias in final linear if present
                            pass
                    # fallback assume 4 coords (two points)
                    out_dim = 4
                    self.model = TinyNet(out_dim).to(device)
                    try:
                        self.model.load_state_dict(state)
                        self.backend = 'torch'
                        print("Loaded iris PyTorch state_dict into TinyNet fallback.")
                    except Exception:
                        # maybe state is {'model': state_dict}
                        if 'model' in state and isinstance(state['model'], dict):
                            self.model.load_state_dict(state['model'])
                            self.backend = 'torch'
                            print("Loaded iris PyTorch model from state['model'].")
                        else:
                            # try loading full object
                            self.model = state
                            self.backend = 'torch'
                            print("Loaded iris PyTorch full model object.")
                else:
                    # state might be a model object
                    self.model = state
                    self.backend = 'torch'
                    print("Loaded iris PyTorch model object from file.")
            except Exception as e:
                print("Failed to load PyTorch model:", e)
                self.model = None
        if self.backend is None:
            print("No iris model loaded (ONNX or PyTorch not found). Iris fallback disabled.")
        else:
            # Preprocessing params
            self.input_size = (64,64)  # model input size expected; change if your model expects other
            import torchvision.transforms as T
            self.transform = T.Compose([T.ToPILImage(), T.Resize(self.input_size), T.ToTensor()])

    def predict(self, eye_crop_bgr):
        """
        Returns list of (x,y) pixel coords in crop coordinates, or [] on failure.
        """
        if self.backend is None:
            return []
        H, W = eye_crop_bgr.shape[:2]
        try:
            if self.backend == 'onnx':
                img = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.input_size)
                img = img.astype(np.float32) / 255.0
                # ONNX model may expect NCHW
                x = np.transpose(img, (2,0,1)).astype(np.float32)[None, ...]
                input_name = self.session.get_inputs()[0].name
                outs = self.session.run(None, {input_name: x})
                out = np.array(outs[0]).reshape(-1)
            elif self.backend == 'torch':
                import torch
                img = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2RGB)
                t = self.transform(img).unsqueeze(0).to(self.device)
                self.model.eval()
                with torch.no_grad():
                    out_t = self.model(t)
                out = out_t.cpu().numpy().reshape(-1)
            else:
                return []
            # Convert normalized coords to pixel coords
            coords = []
            for i in range(0, len(out), 2):
                nx, ny = float(out[i]), float(out[i+1])
                # If values outside [0,1], assume [-1,1] and remap
                if nx < 0 or nx > 1: nx = (nx + 1.0) / 2.0
                if ny < 0 or ny > 1: ny = (ny + 1.0) / 2.0
                px = max(0.0, min(1.0, nx)) * W
                py = max(0.0, min(1.0, ny)) * H
                coords.append((px, py))
            return coords
        except Exception as e:
            # if model errors, return empty
            # print("IrisPredictor.predict error:", e)
            return []

# ----------------- Main behaviour scoring -----------------
def calibrate_neutral(cap, face_mesh, seconds=4):
    print(f"Calibration: look at camera for {seconds} seconds.")
    start = time.time()
    yaws=[]; pitches=[]
    while time.time()-start < seconds:
        ret, frame = cap.read()
        if not ret: continue
        h,w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            try:
                img_pts = get_image_points(lm, w, h)
                focal = w
                cam = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float64)
                dist = np.zeros((4,1))
                ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, img_pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    p,y,r = rotation_vector_to_euler(rvec)
                    yaws.append(y); pitches.append(p)
            except Exception:
                pass
        cv2.putText(frame, "Calibration: look at camera", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow("Calibration")
    if yaws and pitches:
        return float(np.mean(yaws)), float(np.mean(pitches))
    return 0.0, 0.0

def main():
    ensure_dir(OUTPUT_DIR)
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Cannot open camera", CAM_INDEX); return

    mp_face = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # prepare iris predictor (try ONNX then PyTorch)
    iris_pred = IrisPredictor()

    # calibrate
    neutral_yaw, neutral_pitch = calibrate_neutral(cap, face_mesh, seconds=4)
    print(f"Calibration neutral yaw={neutral_yaw:.2f}, pitch={neutral_pitch:.2f}")
    yaw_thresh = DEFAULT_YAW_THRESH; pitch_thresh = DEFAULT_PITCH_THRESH

    # buffers and state
    buf = deque(maxlen=int(WINDOW_SECS * FPS_TARGET) + 6)
    prev_hand_landmarks = []
    last_agg = time.time()
    last_save_ts = 0.0
    last_flag_state = False
    fps = 0.0; prev_time = time.time()

    gaze_prob = 1.0
    gaze_prob_hist = deque(maxlen=MEDIAN_K)

    print("Starting detection. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: continue
        h,w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        fresults = face_mesh.process(rgb)
        hresults = hands.process(rgb)

        frame_gaze_bool = False
        method = "none"
        face_diag = max(w,h)

        if fresults.multi_face_landmarks:
            lm = fresults.multi_face_landmarks[0].landmark
            # 1) Try MediaPipe iris landmarks if present
            try:
                left_iris = None; right_iris = None
                for i in LEFT_IRIS_IDX:
                    if i < len(lm): left_iris = left_iris or []
                    else: left_iris = None; break
                # simpler: check first iris index presence
                if LEFT_IRIS_IDX[0] < len(lm) and RIGHT_IRIS_IDX[0] < len(lm):
                    # extract iris centers from medipipe landmarks (if available)
                    def iris_center_from_indices(indices):
                        pts=[]
                        for i in indices:
                            if i < len(lm):
                                pts.append((lm[i].x*w, lm[i].y*h))
                        return np.array(pts).mean(axis=0) if pts else None
                    left_iris = iris_center_from_indices(LEFT_IRIS_IDX)
                    right_iris = iris_center_from_indices(RIGHT_IRIS_IDX)
                # compute eye boxes
                def eye_box(corner_inds):
                    a = lm[corner_inds[0]]; b = lm[corner_inds[1]]
                    x_min = min(a.x,b.x)*w; x_max = max(a.x,b.x)*w
                    y_min = min(a.y,b.y)*h; y_max = max(a.y,b.y)*h
                    return int(x_min), int(x_max), int(y_min), int(y_max)

                gaze_iris = False
                iris_valid_count = 0
                if left_iris is not None:
                    xmin,xmax,ymin,ymax = eye_box(LEFT_EYE_CORNERS)
                    if xmax-xmin>1 and ymax-ymin>1:
                        cx = (left_iris[0] - (xmin+xmax)/2) / (xmax-xmin)
                        cy = (left_iris[1] - (ymin+ymax)/2) / (ymax-ymin)
                        if abs(cx) <= H_CENTRAL_RATIO/2 and abs(cy) <= V_CENTRAL_RATIO/2:
                            gaze_iris = True
                        iris_valid_count += 1
                        cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(150,200,50),1)
                        cv2.circle(frame,(int(left_iris[0]),int(left_iris[1])),3,(0,255,0) if gaze_iris else (0,0,255),-1)
                if right_iris is not None:
                    xmin2,xmax2,ymin2,ymax2 = eye_box(RIGHT_EYE_CORNERS)
                    if xmax2-xmin2>1 and ymax2-ymin2>1:
                        cx2 = (right_iris[0] - (xmin2+xmax2)/2) / (xmax2-xmin2)
                        cy2 = (right_iris[1] - (ymin2+ymax2)/2) / (ymax2-ymin2)
                        if abs(cx2) <= H_CENTRAL_RATIO/2 and abs(cy2) <= V_CENTRAL_RATIO/2:
                            gaze_iris = gaze_iris or True
                        iris_valid_count += 1
                        cv2.rectangle(frame,(xmin2,ymin2),(xmax2,ymax2),(150,200,50),1)
                        cv2.circle(frame,(int(right_iris[0]),int(right_iris[1])),3,(0,255,0) if gaze_iris else (0,0,255),-1)

                if iris_valid_count > 0:
                    frame_gaze_bool = bool(gaze_iris)
                    method = "mp_iris"
                else:
                    # 2) use model fallback if available
                    if iris_pred.backend is not None:
                        # build eye crop for left eye (use both later)
                        def make_eye_crop(corner_inds, pad=0.15):
                            a = lm[corner_inds[0]]; b = lm[corner_inds[1]]
                            xmin = int(min(a.x,b.x)*w - pad*w); xmax = int(max(a.x,b.x)*w + pad*w)
                            ymin = int(min(a.y,b.y)*h - pad*h); ymax = int(max(a.y,b.y)*h + pad*h)
                            xmin, ymin = max(0, xmin), max(0, ymin)
                            xmax, ymax = min(w-1, xmax), min(h-1, ymax)
                            if xmax - xmin < 4 or ymax - ymin < 4: return None, (0,0)
                            crop = frame[ymin:ymax, xmin:xmax].copy()
                            return crop, (xmin, ymin)
                        left_crop, left_off = make_eye_crop(LEFT_EYE_CORNERS)
                        right_crop, right_off = make_eye_crop(RIGHT_EYE_CORNERS)
                        vote = False; valid = 0
                        if left_crop is not None:
                            coords = iris_pred.predict(left_crop)
                            if coords:
                                valid += 1
                                cx,cy = coords[0]
                                # relative to box center
                                Wc,Hc = left_crop.shape[1], left_crop.shape[0]
                                rx = (cx - Wc/2) / Wc
                                ry = (cy - Hc/2) / Hc
                                if abs(rx) <= H_CENTRAL_RATIO/2 and abs(ry) <= V_CENTRAL_RATIO/2:
                                    vote = True
                                # draw on frame
                                fx,fy = int(left_off[0]+cx), int(left_off[1]+cy)
                                cv2.circle(frame, (fx,fy), 3, (0,255,0) if vote else (0,0,255), -1)
                        if right_crop is not None:
                            coords = iris_pred.predict(right_crop)
                            if coords:
                                valid += 1
                                cx,cy = coords[0]
                                Wc,Hc = right_crop.shape[1], right_crop.shape[0]
                                rx = (cx - Wc/2) / Wc
                                ry = (cy - Hc/2) / Hc
                                if abs(rx) <= H_CENTRAL_RATIO/2 and abs(ry) <= V_CENTRAL_RATIO/2:
                                    vote = vote or True
                                fx,fy = int(right_off[0]+cx), int(right_off[1]+cy)
                                cv2.circle(frame, (fx,fy), 3, (0,255,0) if vote else (0,0,255), -1)
                        if valid > 0:
                            frame_gaze_bool = bool(vote)
                            method = "iris_model"
                        else:
                            # 3) pose fallback
                            try:
                                pts = get_image_points(lm, w, h)
                                focal = w
                                cam = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float64)
                                dist = np.zeros((4,1))
                                ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
                                if ok:
                                    p,y,r = rotation_vector_to_euler(rvec)
                                    frame_gaze_bool = (abs(y - neutral_yaw) <= yaw_thresh) and (abs(p - neutral_pitch) <= pitch_thresh)
                                    method = "pose"
                            except Exception:
                                pass
                    else:
                        # no iris model => pose fallback
                        try:
                            pts = get_image_points(lm, w, h)
                            focal = w
                            cam = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float64)
                            dist = np.zeros((4,1))
                            ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
                            if ok:
                                p,y,r = rotation_vector_to_euler(rvec)
                                frame_gaze_bool = (abs(y - neutral_yaw) <= yaw_thresh) and (abs(p - neutral_pitch) <= pitch_thresh)
                                method = "pose"
                        except Exception:
                            pass

                xs = [p.x for p in lm]; ys = [p.y for p in lm]
                face_diag = math.hypot((max(xs)-min(xs))*w, (max(ys)-min(ys))*h)
            except Exception:
                pass

            mp_drawing.draw_landmarks(frame, fresults.multi_face_landmarks[0], mp_face.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
            cv2.putText(frame, f"method:{method}", (10,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

        # hands
        hand_count = 0
        cur_hands = []
        if hresults.multi_hand_landmarks:
            hand_count = len(hresults.multi_hand_landmarks)
            for hlm in hresults.multi_hand_landmarks:
                cur_hands.append(hlm)
                mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

        # hand motion energy
        raw_motion = 0.0
        if prev_hand_landmarks and cur_hands:
            energies=[]
            for cur in cur_hands:
                cur_pts=np.array([[p.x,p.y,p.z] for p in cur.landmark])
                best=None
                for prev in prev_hand_landmarks:
                    prev_pts=np.array([[p.x,p.y,p.z] for p in prev.landmark])
                    d=np.linalg.norm((cur_pts - prev_pts).reshape(-1,3), axis=1).mean()
                    if best is None or d < best: best = d
                if best is not None: energies.append(best)
            raw_motion = float(np.mean(energies)) if energies else 0.0
        hand_motion_norm = (raw_motion * face_diag) / (face_diag + 1e-6)
        prev_hand_landmarks = cur_hands

        # smoothing: EMA + median
        gaze_val = 1.0 if frame_gaze_bool else 0.0
        gaze_prob = (1-EMA_ALPHA)*gaze_prob + EMA_ALPHA*gaze_val
        gaze_prob_hist.append(1 if gaze_prob > 0.5 else 0)
        if len(gaze_prob_hist) >= MEDIAN_K:
            median_vote = int(np.median(list(gaze_prob_hist)))
            gaze_on = bool(median_vote)
        else:
            gaze_on = gaze_prob > 0.5

        # append to buffer
        buf.append({"t": time.time(), "frame": frame.copy(), "gaze": bool(gaze_on),
                    "hand_present": bool(hand_count>0), "hand_motion": float(hand_motion_norm)})

        # aggregate
        now = time.time()
        if now - last_agg >= AGG_INTERVAL:
            last_agg = now
            frames = list(buf); N = len(frames)
            if N > 0:
                fraction_looking = sum(1 for x in frames if x["gaze"]) / N
                # max contiguous away
                maxc=0; cur=0
                for x in frames:
                    if not x["gaze"]: cur+=1
                    else:
                        if cur>maxc: maxc=cur
                        cur=0
                if cur>maxc: maxc=cur
                max_contig_secs = maxc / max(1, FPS_TARGET)
                hand_presence_frac = sum(1 for x in frames if x["hand_present"]) / N
                hand_motion_avg = sum(x["hand_motion"] for x in frames) / N

                S_gaze = gaze_score(fraction_looking)
                S_away = away_score(max_contig_secs)
                S_hand = hand_score(hand_motion_avg, hand_presence_frac)
                S_behavior = W_G*S_gaze + W_A*S_away + W_H*S_hand

                print(f"[agg] t={time.strftime('%H:%M:%S')}, frac_look={fraction_looking:.2f}, max_away={max_contig_secs:.2f}s, hand_frac={hand_presence_frac:.2f}, hand_motion={hand_motion_avg:.4f} => S_beh={S_behavior:.3f}")

                flagged = (S_behavior < FLAG_BEHAVIOR_THRESH) or (max_contig_secs > MAX_ALLOWED_AWAY)
                if flagged and (not last_flag_state) and (now - last_save_ts > SAVE_COOLDOWN):
                    last_save_ts = now
                    save_clip(frames, OUTPUT_DIR, SAVE_SECONDS, FPS_TARGET)
                last_flag_state = flagged

        # fps
        dt = now - prev_time; prev_time = now
        if dt>0: fps = 0.9*fps + 0.1*(1.0/dt)

        # overlay
        y=30
        def put(s,color=(0,255,0)):
            nonlocal y
            cv2.putText(frame,s,(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2); y+=22
        put(f"FPS:{fps:.1f}")
        put(f"gaze_prob:{gaze_prob:.2f} gaze_on:{int(gaze_on)} method:{method}")
        if buf:
            last=buf[-1]
            put(f"Gaze:{last['gaze']} Hands:{last['hand_present']} HM:{last['hand_motion']:.4f}")
        color = (0,255,0) if (buf and buf[-1]["gaze"]) else (0,0,255)
        cv2.circle(frame,(w-40,40),16,color,-1)

        cv2.imshow("Behaviour Combined - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    face_mesh.close(); hands.close()
    cap.release(); cv2.destroyAllWindows()
    print("Exited.")

if __name__ == "__main__":
    main()
