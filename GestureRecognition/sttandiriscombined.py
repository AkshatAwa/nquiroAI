"""
combined_interview.py

Combines:
 - behaviour/iris detection (MediaPipe + OpenCV) from your main.py
 - speech-to-text via whisper from your testing.py
 - lightweight answer evaluation and CSV logging

Controls (while webcam window focused):
  - 'q' : quit
  - 'r' : start/stop recording candidate audio (saved as wav)
  - 'n' : clear last transcript overlay
  - 'c' : re-run calibration (look at camera)
"""
import os, ctypes
ctypes.CDLL = lambda name, *args, **kwargs: None  # Patch to bypass libc load
import os
import time
import csv
import queue
import threading
import tempfile
import numpy as np
import cv2
import math

# audio
import sounddevice as sd
import soundfile as sf

# whisper
import whisper

# mediapipe
import mediapipe as mp

# ------------------ Load or paste behaviour code pieces ------------------
# For brevity, I reuse the key helpers from your main.py (gaze scoring, pose, iris fallback)
# If you already have a file, you can import functions instead.
# Below are condensed helpers and the IrisPredictor wrapper (keeps compatibility).

# --- Config (small subset pulled from your main) ---
CAM_INDEX = 0
FPS_TARGET = 15
WINDOW_SECS = 8
AGG_INTERVAL = 2.0
OUTPUT_DIR = "flags_combined"
SAVE_SECONDS = 4
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

# PnP and landmark indices (same as your main)
LANDMARK_IDS = {"nose_tip":1,"chin":152,"left_eye_outer":33,"right_eye_outer":280,"left_mouth":61,"right_mouth":291}
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

LEFT_IRIS_IDX = [468,469,470,471]
RIGHT_IRIS_IDX = [473,474,475,476]
LEFT_EYE_CORNERS=(33,133); RIGHT_EYE_CORNERS=(362,263)

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

# ---------------- Iris predictor wrapper (minimal) ----------------
# If you have iris_model.onnx/pth, you can reuse your class. For now we create a minimal stub that returns []
class IrisPredictor:
    def __init__(self):
        self.backend = None
    def predict(self, img):
        return []

# ---------------- Whisper / audio helpers ----------------
MODEL = whisper.load_model("tiny")   # same as testing.py; may take time for first run

AUDIO_DIR = "responses_audio"
TRANSCRIPT_CSV = "transcripts_and_scores.csv"
ensure_dir(AUDIO_DIR)

audio_q = queue.Queue()
recording = {"active": False, "frames": [], "samplerate": 16000, "channels": 1}

def audio_callback(indata, frames, time_info, status):
    if recording["active"]:
        audio_q.put(indata.copy())

def start_recording():
    if recording["active"]:
        return
    recording["active"] = True
    recording["frames"].clear()
    # start input stream in separate thread
    recording["stream"] = sd.InputStream(samplerate=recording["samplerate"], channels=recording["channels"], callback=audio_callback)
    recording["stream"].start()
    print("[audio] recording started")

def stop_recording_and_save():
    if not recording["active"]:
        return None
    recording["active"] = False
    # stop stream
    try:
        recording["stream"].stop()
        recording["stream"].close()
    except Exception:
        pass
    # collect frames
    frames = []
    while not audio_q.empty():
        frames.append(audio_q.get())
    if frames:
        arr = np.concatenate(frames, axis=0)
    else:
        arr = np.zeros((1,recording["channels"]), dtype=np.float32)
    # save to temporary file
    ts = time.strftime("%Y%m%d-%H%M%S")
    wav_path = os.path.join(AUDIO_DIR, f"resp_{ts}.wav")
    sf.write(wav_path, arr, recording["samplerate"])
    print(f"[audio] saved {wav_path}")
    return wav_path

def transcribe_audio(wav_path):
    print("[whisper] transcribing", wav_path)
    try:
        result = MODEL.transcribe(wav_path, fp16=False)  # fp16 may fail on CPU-only
        text = result.get("text","").strip()
        return text
    except Exception as e:
        print("Whisper transcription error:", e)
        return ""

# ---------------- Simple evaluator ----------------
# Lightweight scoring combining keyword matching + fuzzy ratio

from difflib import SequenceMatcher

def similarity(a,b):
    return SequenceMatcher(None, a, b).ratio()

def evaluate_answer(transcript, expected_answers):
    """
    expected_answers: list of acceptable short answers or keywords.
    Returns score in [0,1] and explanation.
    Strategy:
     - If any keyword (word) in expected matches transcript words -> boost
     - Use longest similarity to any expected string as base.
    """
    t = transcript.lower()
    words = set([w.strip(".,!?") for w in t.split()])
    best_sim = 0.0
    for exp in expected_answers:
        sim = similarity(t, exp.lower())
        best_sim = max(best_sim, sim)
    # keyword match
    keyword_hits = 0
    total_keywords = 0
    for exp in expected_answers:
        kws = [w for w in exp.lower().split() if len(w)>3]  # small heuristic
        for kw in kws:
            total_keywords += 1
            if kw in t:
                keyword_hits += 1
    kw_score = (keyword_hits / total_keywords) if total_keywords>0 else 0.0
    # combine
    score = 0.6 * best_sim + 0.4 * kw_score
    score = max(0.0, min(1.0, score))
    explanation = f"sim={best_sim:.2f}, kw={kw_score:.2f} ({keyword_hits}/{total_keywords})"
    return score, explanation

# ---------------- CSV logging ----------------
def append_csv(row):
    first = not os.path.exists(TRANSCRIPT_CSV)
    with open(TRANSCRIPT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if first:
            w.writerow(["timestamp","wav","transcript","score","explanation"])
        w.writerow(row)

# ---------------- Main webcam + behaviour loop (adapted) ----------------
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

def main_loop():
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

    iris_pred = IrisPredictor()

    neutral_yaw, neutral_pitch = calibrate_neutral(cap, face_mesh, seconds=3)
    print(f"Calibration neutral yaw={neutral_yaw:.2f}, pitch={neutral_pitch:.2f}")
    yaw_thresh = DEFAULT_YAW_THRESH; pitch_thresh = DEFAULT_PITCH_THRESH

    buf = []
    prev_hand_landmarks = []
    last_agg = time.time()
    last_save_ts = 0.0
    last_flag_state = False
    fps = 0.0; prev_time = time.time()

    gaze_prob = 1.0
    gaze_prob_hist = []

    last_transcript = ""
    last_score = None
    last_score_expl = ""

    print("Starting detection. Press 'q' to quit, 'r' to start/stop recording audio.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        h,w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        fresults = face_mesh.process(rgb)
        hresults = hands.process(rgb)

        frame_gaze_bool = False
        method = "none"
        face_diag = max(w,h)

        if fresults.multi_face_landmarks:
            lm = fresults.multi_face_landmarks[0].landmark
            # attempt medipipe iris centers (as in your main)
            try:
                left_iris = None; right_iris = None
                if LEFT_IRIS_IDX[0] < len(lm) and RIGHT_IRIS_IDX[0] < len(lm):
                    def iris_center_from_indices(indices):
                        pts=[]
                        for i in indices:
                            if i < len(lm):
                                pts.append((lm[i].x*w, lm[i].y*h))
                        return np.array(pts).mean(axis=0) if pts else None
                    left_iris = iris_center_from_indices(LEFT_IRIS_IDX)
                    right_iris = iris_center_from_indices(RIGHT_IRIS_IDX)

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
                    # fallback pose
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

        # hand motion energy (simple placeholder)
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

        # smoothing & buffer appends
        gaze_val = 1.0 if frame_gaze_bool else 0.0
        gaze_prob = (1-EMA_ALPHA)*gaze_prob + EMA_ALPHA*gaze_val
        gaze_prob_hist.append(1 if gaze_prob > 0.5 else 0)
        if len(gaze_prob_hist) > MEDIAN_K:
            gaze_prob_hist.pop(0)
        if len(gaze_prob_hist) >= MEDIAN_K:
            median_vote = int(np.median(list(gaze_prob_hist)))
            gaze_on = bool(median_vote)
        else:
            gaze_on = gaze_prob > 0.5

        buf.append({"t": time.time(), "frame": frame.copy(), "gaze": bool(gaze_on),
                    "hand_present": bool(hand_count>0), "hand_motion": float(hand_motion_norm)})
        if len(buf) > int(WINDOW_SECS * FPS_TARGET) + 6:
            buf.pop(0)

        # aggregate
        now = time.time()
        if now - last_agg >= AGG_INTERVAL:
            last_agg = now
            frames = list(buf); N = len(frames)
            if N > 0:
                fraction_looking = sum(1 for x in frames if x["gaze"]) / N
                maxc=0; curc=0
                for x in frames:
                    if not x["gaze"]: curc+=1
                    else:
                        if curc>maxc: maxc=curc
                        curc=0
                if curc>maxc: maxc=curc
                max_contig_secs = maxc / max(1, FPS_TARGET)
                hand_presence_frac = sum(1 for x in frames if x["hand_present"]) / N
                hand_motion_avg = sum(x["hand_motion"] for x in frames) / N

                S_gaze = gaze_score(fraction_looking)
                S_away = away_score(max_contig_secs)
                S_hand = hand_score(hand_motion_avg, hand_presence_frac)
                S_behavior = W_G*S_gaze + W_A*S_away + W_H*S_hand

                # print small log
                # print(f"[agg] frac_look={fraction_looking:.2f}, max_away={max_contig_secs:.2f}s, hand_frac={hand_presence_frac:.2f} => S_beh={S_behavior:.3f}")
                flagged = (S_behavior < FLAG_BEHAVIOR_THRESH) or (max_contig_secs > MAX_ALLOWED_AWAY)
                if flagged and (not last_flag_state) and (now - last_save_ts > SAVE_COOLDOWN):
                    last_save_ts = now
                    save_clip(frames, OUTPUT_DIR, SAVE_SECONDS, FPS_TARGET)
                last_flag_state = flagged

        # fps
        dt = now - prev_time; prev_time = now
        if dt>0: fps = 0.9*fps + 0.1*(1.0/dt)

        # overlay UI
        y=30
        def put(s,color=(0,255,0)):
            nonlocal y
            cv2.putText(frame,s,(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2); y+=22
        put(f"FPS:{fps:.1f}")
        put(f"GazeProb:{gaze_prob:.2f} GazeOn:{int(gaze_on)} Method:{method}")
        if buf:
            last = buf[-1]
            put(f"Gaze:{last['gaze']} Hands:{last['hand_present']} HM:{last['hand_motion']:.4f}")
        color = (0,255,0) if (buf and buf[-1]["gaze"]) else (0,0,255)
        cv2.circle(frame,(w-40,40),16,color,-1)

        # show last transcript and score
        if last_transcript:
            cv2.putText(frame, "Transcript:", (10,h-120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)
            lines = []
            t = last_transcript
            # break long transcript into lines
            while len(t)>60:
                lines.append(t[:60])
                t = t[60:]
            lines.append(t)
            ly = h-100
            for L in lines:
                cv2.putText(frame, L, (10, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200),1)
                ly += 20
            if last_score is not None:
                cv2.putText(frame, f"Score: {last_score:.2f} ({last_score_expl})", (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)

        cv2.imshow("Combined Interview - press q to quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            # toggle recording
            if not recording["active"]:
                start_recording()
            else:
                wav = stop_recording_and_save()
                if wav:
                    # transcribe synchronously (could be moved to a thread)
                    txt = transcribe_audio(wav)
                    # evaluate: you can set expected answers for each question; for demo use placeholder list
                    expected = ["I have experience with Python, JavaScript and React", "I have experience in React and backend"]
                    score, expl = evaluate_answer(txt, expected)
                    append_csv([time.strftime("%Y-%m-%d %H:%M:%S"), wav, txt, f"{score:.3f}", expl])
                    last_transcript = txt
                    last_score = score
                    last_score_expl = expl
        elif key == ord('n'):
            last_transcript = ""; last_score = None; last_score_expl = ""

    face_mesh.close(); hands.close()
    cap.release(); cv2.destroyAllWindows()
    print("Exited.")

if __name__ == "__main__":
    main_loop()
