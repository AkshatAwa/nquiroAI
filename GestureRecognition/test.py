# interview_backend_faster_whisper.py
import base64
import io
import os
import time
import uuid
import json
import queue
import threading
import csv
import subprocess
import shutil
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import cv2
import mediapipe as mp

# faster-whisper
from faster_whisper import WhisperModel

# ---------------- Config ----------------
AUDIO_DIR = "responses_audio"
TRANSCRIPT_CSV = "transcripts_and_scores.csv"
ensure_dirs = [AUDIO_DIR]
for d in ensure_dirs:
    os.makedirs(d, exist_ok=True)

# Whisper model: CPU mode by default. Change device="cuda" if you have GPU/CUDA configured.
WHISPER_MODEL_NAME = "tiny"  # change to small/base/medium if you have resources
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8_float32"  # good CPU option; change if needed

# MediaPipe constants (we use lightweight per-connection FaceMesh)
MP_FACE = mp.solutions.face_mesh

# FastAPI app
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------- Globals ----------------
clients: Dict[str, WebSocket] = {}
clients_lock = threading.Lock()

transcribe_q = queue.Queue()
stop_workers = threading.Event()
csv_lock = threading.Lock()

# UI state variables (worker updates)
last_transcripts_by_client: Dict[str, str] = {}

# ---------------- Load Whisper (faster-whisper) ----------------
print("[startup] loading faster-whisper model. This may take time...")
MODEL = WhisperModel(WHISPER_MODEL_NAME, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
print("[startup] model loaded.")

# ---------------- Helpers ----------------
from difflib import SequenceMatcher
def similarity(a,b):
    return SequenceMatcher(None, a, b).ratio()

def evaluate_answer(transcript, expected_answers):
    t = (transcript or "").lower()
    best_sim = 0.0
    for exp in expected_answers:
        sim = similarity(t, exp.lower())
        if sim > best_sim: best_sim = sim
    keyword_hits = 0
    total_keywords = 0
    for exp in expected_answers:
        kws = [w for w in exp.lower().split() if len(w)>3]
        for kw in kws:
            total_keywords += 1
            if kw in t:
                keyword_hits += 1
    kw_score = (keyword_hits / total_keywords) if total_keywords>0 else 0.0
    score = 0.6 * best_sim + 0.4 * kw_score
    score = max(0.0, min(1.0, score))
    explanation = f"sim={best_sim:.2f}, kw={kw_score:.2f} ({keyword_hits}/{total_keywords})"
    return score, explanation

def append_csv(row):
    first = not os.path.exists(TRANSCRIPT_CSV)
    with csv_lock:
        with open(TRANSCRIPT_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if first:
                w.writerow(["timestamp","wav","transcript","score","explanation"])
            w.writerow(row)

# ffmpeg helper
def ffmpeg_available():
    return shutil.which("ffmpeg") is not None

def transcode_to_wav(input_path, target_sr=16000):
    """
    Transcode input audio/video file to a single-channel WAV at target_sr.
    Returns wav_path (string) on success or None on failure.
    """
    if not ffmpeg_available():
        print("[worker] ffmpeg not found in PATH")
        return None

    base = os.path.splitext(os.path.basename(input_path))[0]
    wav_name = f"{base}.wav"
    wav_path = os.path.join(os.path.dirname(input_path), wav_name)

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", str(target_sr),
        "-ac", "1",
        "-vn",
        "-f", "wav",
        wav_path
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 100:
            return wav_path
        else:
            print("[worker] transcode produced empty file:", wav_path)
            try:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception:
                pass
            return None
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else str(e)
        print("[worker] ffmpeg transcode failed:", stderr)
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass
        return None

# ---------------- Transcription (worker) ----------------
def transcribe_with_faster_whisper(wav_path):
    """Return combined transcript (string) for file at wav_path. Uses MODEL loaded above."""
    try:
        segments, info = MODEL.transcribe(wav_path, beam_size=5)
        texts = [seg.text for seg in segments]
        return " ".join(texts).strip()
    except Exception as e:
        print("[worker] faster-whisper error:", e)
        return ""

def transcription_worker():
    while not stop_workers.is_set():
        try:
            item = transcribe_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            transcribe_q.task_done()
            break

        input_path, client_id = item
        print("[worker] got", input_path)

        # sanity checks
        if not os.path.exists(input_path) or os.path.getsize(input_path) < 50:
            print("[worker] invalid or empty input file:", input_path)
            transcribe_q.task_done()
            continue

        # Attempt transcode to wav
        wav_path = transcode_to_wav(input_path, target_sr=16000)
        if wav_path is None:
            # fallback: attempt to transcribe original (may fail)
            print("[worker] transcode failed; attempting to transcribe original file")
            text = transcribe_with_faster_whisper(input_path)
        else:
            print("[worker] transcribing wav:", wav_path)
            text = transcribe_with_faster_whisper(wav_path)
            # cleanup wav to save space
            try:
                os.remove(wav_path)
            except Exception:
                pass

        expected = [ "I have experience with Python, JavaScript and React",
                     "I have experience in React and backend" ]
        score, expl = evaluate_answer(text, expected)
        append_csv([time.strftime("%Y-%m-%d %H:%M:%S"), input_path, text, f"{score:.3f}", expl])

        last_transcripts_by_client[client_id] = text

        # send result to websocket if present
        with clients_lock:
            ws = clients.get(client_id)
        if ws:
            payload = {"type":"transcript", "text": text, "score": score, "explanation": expl, "wav": os.path.basename(input_path)}
            try:
                import asyncio
                # send_text is async; run it in a new loop to deliver from worker thread
                asyncio.run(ws.send_text(json.dumps(payload)))
            except Exception as e:
                print("[worker] failed send to websocket:", e)

        transcribe_q.task_done()

worker_thread = threading.Thread(target=transcription_worker, daemon=True)
worker_thread.start()

# ---------------- WebSocket: frames & detection ----------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = str(uuid.uuid4())
    # create mediapipe FaceMesh for this connection
    face_mesh = MP_FACE.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
    await websocket.send_text(json.dumps({"type":"welcome", "client_id":client_id}))
    with clients_lock:
        clients[client_id] = websocket
    print("[ws] client connected", client_id)
    try:
        while True:
            msg = await websocket.receive_text()
            # expecting JSON messages, type 'frame' with base64 jpeg data
            try:
                data = json.loads(msg)
            except Exception:
                continue
            if data.get("type") == "frame" and data.get("img"):
                b64 = data["img"].split(",")[-1]
                frame_bytes = base64.b64decode(b64)
                arr = np.frombuffer(frame_bytes, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                # Mediapipe detection
                h,w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)
                gaze_on = False
                method = "none"
                try:
                    if res.multi_face_landmarks:
                        lm = res.multi_face_landmarks[0].landmark
                        # simple iris centroid heuristic: check left iris
                        def iris_center(indices):
                            pts=[]
                            for i in indices:
                                if i < len(lm):
                                    pts.append((lm[i].x*w, lm[i].y*h))
                            return np.array(pts).mean(axis=0) if pts else None
                        left_idx = [468,469,470,471]
                        right_idx = [473,474,475,476]
                        left_c = iris_center(left_idx)
                        right_c = iris_center(right_idx)
                        if left_c is not None:
                            lx = int(lm[33].x*w); rx = int(lm[133].x*w)
                            cx = (left_c[0] - (lx+rx)/2) / max(1, rx-lx)
                            if abs(cx) < 0.25:
                                gaze_on = True; method = "mp_iris"
                        if not gaze_on and right_c is not None:
                            lx = int(lm[362].x*w); rx = int(lm[263].x*w)
                            cx2 = (right_c[0] - (lx+rx)/2) / max(1, rx-lx)
                            if abs(cx2) < 0.25:
                                gaze_on = True; method = "mp_iris"
                except Exception:
                    pass
                # reply detection
                payload = {"type":"detection", "gaze": gaze_on, "method": method}
                await websocket.send_text(json.dumps(payload))
            # can handle other message types here
    except WebSocketDisconnect:
        print("[ws] client disconnected", client_id)
    finally:
        try:
            face_mesh.close()
        except Exception:
            pass
        with clients_lock:
            if client_id in clients:
                del clients[client_id]

# ---------------- Audio upload endpoint ----------------
@app.post("/upload_audio")
async def upload_audio(client_id: str = Form(...), file: UploadFile = File(...)):
    try:
        ts = time.strftime("%Y%m%d-%H%M%S")
        filename = f"resp_{ts}_{uuid.uuid4().hex[:6]}.webm"
        path = os.path.join(AUDIO_DIR, filename)
        contents = await file.read()
        with open(path, "wb") as f:
            f.write(contents)
        # Enqueue path + client id
        transcribe_q.put((path, client_id))
        print("[upload] saved", path)
        return JSONResponse({"status":"queued", "wav": os.path.basename(path)})
    except Exception as e:
        return JSONResponse({"status":"error", "detail": str(e)}, status_code=500)

# ---------------- Shutdown handling ----------------
import atexit
def _shutdown():
    print("[shutdown] stopping worker")
    stop_workers.set()
    try:
        transcribe_q.put(None)
    except Exception:
        pass
atexit.register(_shutdown)

# Allow running directly with python filename
if __name__ == "__main__":
    import uvicorn
    # run the FastAPI app in the current process (blocking)
    uvicorn.run("test:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
