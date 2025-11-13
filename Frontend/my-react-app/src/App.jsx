// src/App.jsx
import React, { useEffect, useRef, useState } from "react";
import FullScreenAvatarProp from "./components/modellipsync/modellipsync";
import CandidatePanel from "./components/candidate/candidate";
import "./App.css";

export default function App() {
  const [interviewerText, setInterviewerText] = useState(
    "Hello — I will ask you a few questions. Please introduce yourself."
  );
  const [candidateText, setCandidateText] = useState("Candidate answers will appear here...");
  const [isTyping, setIsTyping] = useState(false);
  const [candidateVisible, setCandidateVisible] = useState(false);

  // only interviewer audio is passed to avatar (so only interviewer lipsyncs)
  const [interviewerAudio, setInterviewerAudio] = useState(null);
  // candidate audio plays directly (if you want later)
  const candidateAudioRef = useRef(null);

  // candidate video permission / active state (optional: used to display status)
  const [cameraActive, setCameraActive] = useState(false);

  const candidateRefBox = useRef(null);
  const interviewerRefBox = useRef(null);

  // simulate interviewer text + audio
  const simulateInterviewer = () => {
    const text = "Interviewer: Tell me about a recent project where you used React and Three.js.";
    setIsTyping(true);
    setInterviewerText("");
    // stop other audio
    stopAllAudio();
    setTimeout(() => {
      setInterviewerText(text);
      setIsTyping(false);
      setInterviewerAudio("/interviewer_resp.wav"); // make sure this exists in public/
      // clear after a while (optional)
      // setTimeout(() => setInterviewerAudio(null), 9000);
    }, 150);
  };

  // simulate candidate text + (optional) audio played directly
  const simulateCandidate = () => {
    const text = "Candidate: I built an interactive 3D viewer using React Three Fiber and deployed it on Vercel.";
    setCandidateVisible(false);
    setCandidateText("");
    // stop interviewer audio
    stopAllAudio();
    setTimeout(() => {
      setCandidateText(text);
      setCandidateVisible(true);
      // play candidate audio directly (NOT passed to avatar) if you have one
      // const a = new Audio("/resp_20251110-195404.wav");
      // a.play().catch(() => {});
      // candidateAudioRef.current = a;
    }, 250);
  };

  const stopAllAudio = () => {
    setInterviewerAudio(null);
    try {
      if (candidateAudioRef.current) {
        candidateAudioRef.current.pause();
        candidateAudioRef.current.currentTime = 0;
      }
    } catch (e) {}
    candidateAudioRef.current = null;
  };

  useEffect(() => {
    if (candidateRefBox.current) candidateRefBox.current.scrollTop = candidateRefBox.current.scrollHeight;
  }, [candidateText]);

  useEffect(() => {
    if (interviewerRefBox.current) interviewerRefBox.current.scrollTop = interviewerRefBox.current.scrollHeight;
  }, [interviewerText]);

  return (
    <div className="app-root">
      <div className="columns">
        {/* LEFT: Interviewer / Avatar */}
        <div className="column interviewer-col">
          <div className="panel-title">Interviewer</div>
          <div className="avatar-wrap">
            <FullScreenAvatarProp
              avatarUrl="https://models.readyplayer.me/691079f1d28f9f8e50c6e261.glb"
              bgImage="/model/blurred-image-of-an-office-space.webp"
              modelScale={5.0}
              modelYOffset={-4.1}
              faceTilt={-0.6}
              audioUrl={interviewerAudio}
            />
          </div>

          <div className="transcript small" role="region" aria-label="Interviewer transcript">
            <div className="box-title">Interviewer</div>
            <div className={`box-body ${isTyping ? "typing" : ""}`} ref={interviewerRefBox}>
              {interviewerText}
            </div>
          </div>

          <div className="controls">
            <button onClick={simulateInterviewer}>Simulate Interviewer</button>
            <button onClick={simulateCandidate}>Simulate Candidate</button>
            <button onClick={stopAllAudio}>Stop Audio</button>
          </div>
        </div>

        {/* RIGHT: Candidate (video / iris detection area) */}
        <div className="column candidate-col">
          <div className="panel-title">Candidate</div>

          <CandidatePanel
            onCameraActive={(active) => setCameraActive(active)}
            // processFrame is called with (videoElement, canvasContext, frameImageData)
            // Replace processFrame implementation inside CandidatePanel or pass your model function here
            processFrame={async (videoEl, overlayCtx, imageData) => {
              // Placeholder — your iris/face model code will go here.
              // Example:
              // const results = await myIrisModel.estimate(videoEl);
              // draw landmarks on overlayCtx
            }}
          />

          <div className={`transcript small ${candidateVisible ? "fade-in" : "hidden"}`} role="region" aria-label="Candidate transcript">
            <div className="box-title">Candidate</div>
            <div className="box-body" ref={candidateRefBox}>
              {candidateText}
            </div>
          </div>

          <div className="camera-status">
            Camera: {cameraActive ? <span className="status-on">Active</span> : <span className="status-off">Inactive</span>}
          </div>
        </div>
      </div>
    </div>
  );
}
