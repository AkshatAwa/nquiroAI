// src/components/CandidatePanel.jsx
import React, { useEffect, useRef, useState } from "react";

/**
 * CandidatePanel — Canvas-driven video + WebSocket + reliable WAV recording (WebAudio -> PCM16 WAV)
 *
 * Props:
 *  - processFrame(video, ctx, imageData): optional async frame processor
 *  - onCameraActive(active: boolean)
 *  - onTranscript(text: string)
 *  - serverUrl: backend base URL (default http://localhost:8000)
 *
 * Behavior:
 *  - Streams compressed JPEG frames (~5 FPS) to backend WebSocket /ws
 *  - Records audio using WebAudio, encodes WAV PCM16 @16kHz, uploads to /upload_audio
 *  - Receives detection & transcript messages over WebSocket
 */
export default function CandidatePanel({
  processFrame = null,
  onCameraActive = () => {},
  onTranscript = () => {},
  serverUrl = "http://localhost:8000",
}) {
  // video/canvas refs
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(null);

  // websocket, client id
  const wsRef = useRef(null);
  const clientIdRef = useRef(null);

  // recording refs
  const audioContextRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const processorRef = useRef(null);
  const recordingBuffersRef = useRef([]);
  const recordingLenRef = useRef(0);

  const lastSentRef = useRef(0);

  // state
  const [cameraStarted, setCameraStarted] = useState(false);
  const [facingMode, setFacingMode] = useState("user");
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [wsState, setWsState] = useState("disconnected");

  // constants
  const targetSampleRate = 16000;
  const wsUrl = serverUrl.replace(/^http/, "ws").replace(/\/$/, "") + "/ws";
  const uploadUrl = serverUrl.replace(/\/$/, "") + "/upload_audio";

  /* ------------------ WebSocket ------------------ */
  const connectWebsocket = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;
    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      setWsState("connecting");
      ws.onopen = () => {
        setWsState("connected");
        console.log("[ws] open");
      };
      ws.onclose = () => {
        setWsState("disconnected");
        clientIdRef.current = null;
        console.log("[ws] close");
      };
      ws.onerror = (e) => {
        setWsState("error");
        console.warn("[ws] error", e);
      };
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.type === "welcome" && data.client_id) {
            clientIdRef.current = data.client_id;
            console.log("[ws] client_id:", clientIdRef.current);
          } else if (data.type === "detection") {
            // optional - parent can use processFrame for local overlays
          } else if (data.type === "transcript") {
            const t = data.text || "";
            setTranscript(t);
            onTranscript(t);
          }
        } catch (err) {
          console.warn("[ws] invalid message", err);
        }
      };
    } catch (err) {
      setWsState("error");
      console.warn("[ws] connect failed", err);
    }
  };

  const disconnectWebsocket = () => {
    try {
      if (wsRef.current) {
        try { wsRef.current.close(); } catch (e) {}
        wsRef.current = null;
      }
    } catch (e) {}
    setWsState("disconnected");
    clientIdRef.current = null;
  };

  /* ------------------ Camera / Canvas ------------------ */
  const startCamera = async () => {
    try {
      await stopCamera();
      const constraints = {
        audio: true,
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode },
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraStarted(true);
      onCameraActive(true);
      connectWebsocket();
      lastSentRef.current = 0;
      rafRef.current = requestAnimationFrame(frameLoop);
    } catch (err) {
      console.error("Camera start failed:", err);
      setCameraStarted(false);
      onCameraActive(false);
    }
  };

  const stopCamera = async () => {
    try {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      if (videoRef.current) {
        try { videoRef.current.pause(); } catch (e) {}
        try { videoRef.current.srcObject = null; } catch (e) {}
      }
      if (streamRef.current) {
        try { streamRef.current.getTracks().forEach(t => t.stop()); } catch (e) {}
        streamRef.current = null;
      }
      // stop recording gracefully if active
      if (processorRef.current) {
        try { await stopRecording(); } catch (e) {}
      }
      setCameraStarted(false);
      onCameraActive(false);
      clearCanvas();
      disconnectWebsocket();
    } catch (err) {
      console.warn("stopCamera error", err);
    }
  };

  const flipCamera = async () => {
    setFacingMode(prev => (prev === "user" ? "environment" : "user"));
    await stopCamera();
    setTimeout(() => startCamera(), 200);
  };

  const clearCanvas = () => {
    try {
      const c = overlayRef.current;
      if (!c) return;
      const ctx = c.getContext("2d");
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.fillStyle = "#111";
      ctx.fillRect(0, 0, c.width, c.height);
    } catch (e) {}
  };

  const frameLoop = async () => {
    try {
      const video = videoRef.current;
      const overlay = overlayRef.current;
      if (!video || video.readyState < 2 || !overlay) {
        rafRef.current = requestAnimationFrame(frameLoop);
        return;
      }
      const ctx = overlay.getContext("2d");
      if (overlay.width !== video.videoWidth || overlay.height !== video.videoHeight) {
        overlay.width = video.videoWidth || overlay.clientWidth || 640;
        overlay.height = video.videoHeight || overlay.clientHeight || 480;
        overlay.style.width = "100%";
        overlay.style.height = "100%";
      }
      const vw = video.videoWidth, vh = video.videoHeight, cw = overlay.width, ch = overlay.height;
      const scale = Math.max(cw / vw, ch / vh);
      const sw = cw / scale, sh = ch / scale;
      const sx = Math.max(0, (vw - sw) / 2);
      const sy = Math.max(0, (vh - sh) / 2);

      ctx.clearRect(0, 0, cw, ch);
      ctx.drawImage(video, sx, sy, sw, sh, 0, 0, cw, ch);

      if (processFrame) {
        try {
          const imageData = ctx.getImageData(0, 0, cw, ch);
          Promise.resolve(processFrame(video, ctx, imageData)).catch(e => console.warn("processFrame error", e));
        } catch (e) {
          // getImageData can fail in some contexts
        }
      }

      const now = performance.now();
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && now - lastSentRef.current > 180) {
        try {
          const dataUrl = overlay.toDataURL("image/jpeg", 0.6);
          wsRef.current.send(JSON.stringify({ type: "frame", img: dataUrl }));
          lastSentRef.current = now;
        } catch (e) {}
      }

      rafRef.current = requestAnimationFrame(frameLoop);
    } catch (err) {
      console.warn("frameLoop error:", err);
      rafRef.current = requestAnimationFrame(frameLoop);
    }
  };

  /* ------------------ WebAudio -> WAV helpers ------------------ */
  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) view.setUint8(offset + i, string.charCodeAt(i));
  }

  function downsampleBuffer(buffer, srcRate, dstRate) {
    if (dstRate === srcRate) return buffer;
    const sampleRateRatio = srcRate / dstRate;
    const newLength = Math.round(buffer.length / sampleRateRatio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
      let accum = 0, count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
        accum += buffer[i];
        count++;
      }
      result[offsetResult] = count ? accum / count : 0;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }
    return result;
  }

  function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return new Blob([view], { type: 'audio/wav' });
  }

  /* ------------------ Recording flow ------------------ */
  const startRecording = async () => {
    if (!streamRef.current) {
      console.warn("No media stream to record");
      return;
    }
    if (processorRef.current) {
      console.warn("Already recording");
      return;
    }
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const ac = new AudioCtx();
    audioContextRef.current = ac;
    const source = ac.createMediaStreamSource(streamRef.current);
    sourceNodeRef.current = source;

    const bufferSize = 4096;
    const processor = ac.createScriptProcessor(bufferSize, source.channelCount, 1);
    processorRef.current = processor;

    recordingBuffersRef.current = [];
    recordingLenRef.current = 0;

    processor.onaudioprocess = (ev) => {
      try {
        const inputBuffer = ev.inputBuffer;
        const chCount = inputBuffer.numberOfChannels;
        const inputData = new Float32Array(inputBuffer.length);
        if (chCount === 1) {
          inputBuffer.copyFromChannel(inputData, 0);
        } else {
          // average channels to mono
          const temp = new Float32Array(inputBuffer.length);
          for (let ch = 0; ch < chCount; ch++) {
            inputBuffer.copyFromChannel(temp, ch);
            for (let i = 0; i < inputData.length; i++) inputData[i] = (inputData[i] || 0) + temp[i];
          }
          for (let i = 0; i < inputData.length; i++) inputData[i] = inputData[i] / chCount;
        }
        recordingBuffersRef.current.push(inputData);
        recordingLenRef.current += inputData.length;
      } catch (e) {
        // ignore occasional errors
      }
    };

    // connect nodes - some browsers require processor connected to destination for onaudioprocess to run
    source.connect(processor);
    processor.connect(ac.destination);

    setRecording(true);
    console.log("Recording started (WebAudio -> WAV)");
  };

  const stopRecording = async () => {
    try {
      if (!processorRef.current || !audioContextRef.current) {
        setRecording(false);
        return;
      }
      // disconnect
      try { processorRef.current.disconnect(); } catch(e) {}
      try { sourceNodeRef.current.disconnect(); } catch(e) {}

      const ac = audioContextRef.current;
      const srcSampleRate = ac.sampleRate;

      // merge buffers
      let merged = new Float32Array(recordingLenRef.current);
      let offset = 0;
      for (let i = 0; i < recordingBuffersRef.current.length; i++) {
        merged.set(recordingBuffersRef.current[i], offset);
        offset += recordingBuffersRef.current[i].length;
      }

      // downsample to 16k
      let downsampled = merged;
      if (srcSampleRate !== targetSampleRate) {
        downsampled = downsampleBuffer(merged, srcSampleRate, targetSampleRate);
      }

      // encode WAV
      const wavBlob = encodeWAV(downsampled, targetSampleRate);

      // cleanup AudioContext
      try { await ac.close(); } catch(e) {}
      audioContextRef.current = null;
      sourceNodeRef.current = null;
      processorRef.current = null;
      recordingBuffersRef.current = [];
      recordingLenRef.current = 0;

      // NOTE: Removed automatic playback to avoid echo/repeat

      // upload to backend
      const fd = new FormData();
      fd.append("client_id", clientIdRef.current || "");
      fd.append("file", wavBlob, "resp.wav");
      try {
        const res = await fetch(uploadUrl, { method: "POST", body: fd });
        const json = await res.json();
        console.log("[upload] response", json);
      } catch (err) {
        console.warn("[upload] failed", err);
      }
    } catch (err) {
      console.warn("stopRecording error", err);
    } finally {
      setRecording(false);
    }
  };

  const toggleRecording = () => {
    if (recording) stopRecording();
    else startRecording();
  };

  /* ------------------ Cleanup on unmount ------------------ */
  useEffect(() => {
    return () => {
      try {
        if (rafRef.current) cancelAnimationFrame(rafRef.current);
      } catch (e) {}
      try {
        if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
      } catch (e) {}
      try { if (videoRef.current) { videoRef.current.pause(); videoRef.current.srcObject = null; } } catch(e){}
      try { if (processorRef.current) processorRef.current.disconnect(); } catch(e){}
      try { if (audioContextRef.current) audioContextRef.current.close(); } catch(e){}
      try { if (wsRef.current) wsRef.current.close(); } catch(e){}
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ------------------ Render ------------------ */
  return (
    <div className="candidate-panel" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <video ref={videoRef} playsInline muted style={{ display: "none" }} />
      <div className="video-container-canvas" style={{ position: "relative", width: "100%", aspectRatio: "16/9", background: "#111" }}>
        <canvas ref={overlayRef} className="overlay-canvas-visible" style={{ width: "100%", height: "100%", display: "block" }} />
        {!cameraStarted && (
          <div className="camera-placeholder" style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", color: "#ddd" }}>
            <p>Camera inactive</p>
          </div>
        )}
      </div>

      <div className="candidate-controls" style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <button onClick={startCamera} disabled={cameraStarted}>Start</button>
        <button onClick={stopCamera} disabled={!cameraStarted}>Stop</button>
        <button onClick={flipCamera}>Flip</button>
        <button onClick={toggleRecording} style={{ marginLeft: 12 }}>{recording ? "Stop Recording" : "Start Recording"}</button>

        <div style={{ marginLeft: 8, fontSize: 13 }}>
          <div>WS: {wsState}</div>
          <div>client_id: {clientIdRef.current || "(none)"}</div>
        </div>
      </div>

      {/* Transcript box: shows backend transcript (no duplicates, no audio playback) */}
      <div style={{ marginTop: 8, width: "100%" }}>
        <label style={{ fontSize: 13, color: "#666" }}>Transcript</label>
        <textarea
          value={transcript}
          readOnly
          placeholder="Your spoken words will appear here after you stop recording..."
          rows={4}
          style={{
            width: "100%",
            resize: "vertical",
            padding: 8,
            marginTop: 6,
            borderRadius: 6,
            border: "1px solid #333",
            background: "#0f0f0f",
            color: "#fff",
            fontSize: 14,
          }}
        />
      </div>
    </div>
  );
}
