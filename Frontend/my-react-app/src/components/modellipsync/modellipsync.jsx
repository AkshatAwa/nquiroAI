// src/components/modellipsync/modellipsync.jsx
import React, { Suspense, useRef, useEffect, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { useGLTF, useTexture } from "@react-three/drei";

/* ---------- hide irrelevant body parts ---------- */
function hideUnwantedParts(scene) {
  if (!scene) return;
  const patterns = [
    /hand/i, /arm/i, /forearm/i, /leg/i, /thigh/i, /foot/i, /toe/i,
    /hip/i, /pelvis/i, /spine/i, /torso/i, /shoulder/i
  ];
  scene.traverse((obj) => {
    if (obj.isMesh || obj.isBone) {
      const name = obj.name?.toLowerCase() || "";
      if (patterns.some((p) => name.match(p))) obj.visible = false;
    }
  });
}

/* ---------- Avatar head: finds mouth morphs automatically and drives them ---------- */
function AvatarHead({
  url,
  analyser = null,
  mouthOpen = 0,
  modelScale = 5.0,
  modelYOffset = -4.1,
  faceTilt = -0.6,
}) {
  const { scene } = useGLTF(url);
  const group = useRef();

  // We'll store for each mesh that supports morphs: { meshObj, mouthIndex }
  const morphMeshesRef = useRef([]);
  const jawBoneRef = useRef(null);
  const smoothRef = useRef(0);

  useEffect(() => {
    if (!scene) return;
    hideUnwantedParts(scene);

    morphMeshesRef.current = [];
    jawBoneRef.current = null;

    // helper to test name for mouth-like morphs
    const mouthNameRegex = /mouth|jaw|vrc\.v_|aa|ah|ou|oh|ee|ih/i;

    scene.traverse((obj) => {
      // collect meshes that have morphTargetInfluences
      if (obj.morphTargetInfluences && obj.morphTargetInfluences.length) {
        // Try to find a good morph index from available dictionary (if present)
        let chosenIndex = null;
        const dict = obj.morphTargetDictionary || null;

        if (dict) {
          // look for exact mouth-like names
          const nameEntries = Object.entries(dict);
          // prefer explicit names
          for (const [name, idx] of nameEntries) {
            const lname = name.toLowerCase();
            if (lname.includes("mouthopen") || lname.includes("mouth_open") || lname.includes("jawopen") || lname.includes("jaw_open")) {
              chosenIndex = idx;
              break;
            }
          }
          // otherwise choose any that matches mouth-ish regex
          if (chosenIndex === null) {
            for (const [name, idx] of nameEntries) {
              if (mouthNameRegex.test(name)) {
                chosenIndex = idx;
                break;
              }
            }
          }
        }

        // If still null, as a fallback use index 0 (commonly mouthOpen in many RPM exports)
        if (chosenIndex === null) chosenIndex = 0;

        morphMeshesRef.current.push({ mesh: obj, index: chosenIndex });
      }

      // find a jaw bone fallback if available
      if (obj.isBone) {
        const nm = (obj.name || "").toLowerCase();
        if (nm.includes("jaw") || nm.includes("lowerjaw") || nm.includes("jawbone") || nm.includes("mandible")) {
          jawBoneRef.current = obj;
        }
      }
    });

    console.log("Found morph-capable meshes:", morphMeshesRef.current.length, "jaw bone:", jawBoneRef.current?.name || null);
    // Also print sample dicts for debugging
    morphMeshesRef.current.slice(0,4).forEach((m,i) => {
      console.log(" morph mesh", i, "name:", m.mesh.name, "chosenIndex:", m.index, "dict:", m.mesh.morphTargetDictionary || "no-dict");
    });

  }, [scene]);

  useFrame((state) => {
    if (!group.current) return;
    const t = state.clock.getElapsedTime();

    // subtle idle movement
    group.current.rotation.y = Math.sin(t * 0.05) * 0.005;
    group.current.rotation.x = faceTilt;
    group.current.position.y = modelYOffset;

    // compute mouthValue from analyser (RMS -> 0..1) or fallback mouthOpen prop
    let mouthValue = mouthOpen || 0;
    if (analyser) {
      try {
        const data = new Uint8Array(analyser.fftSize);
        analyser.getByteTimeDomainData(data);
        let sum = 0;
        for (let i = 0; i < data.length; i++) {
          const v = (data[i] - 128) / 128;
          sum += v * v;
        }
        const rms = Math.sqrt(sum / data.length);
        const raw = Math.min(1, rms * 6.0); // tuning multiplier
        smoothRef.current = smoothRef.current * 0.82 + raw * 0.18;
        mouthValue = smoothRef.current;
      } catch (e) {
        // ignore analyser errors
      }
    }

    // strength multiplier — increase if the mouth movement is too subtle
    const strengthMult = 2.5;

    const applied = Math.max(0, Math.min(1, mouthValue * strengthMult));

    // apply to all morph meshes we found
    for (const mm of morphMeshesRef.current) {
      try {
        if (mm.mesh.morphTargetInfluences && typeof mm.index === "number" && mm.index < mm.mesh.morphTargetInfluences.length) {
          mm.mesh.morphTargetInfluences[mm.index] = applied;
        }
      } catch (e) {
        // ignore per-mesh errors
      }
    }

    // if no morph meshes found, fallback to jaw bone rotation
    if (morphMeshesRef.current.length === 0 && jawBoneRef.current) {
      if (!("origRot" in jawBoneRef.current.userData)) jawBoneRef.current.userData.origRot = jawBoneRef.current.rotation.x;
      const base = jawBoneRef.current.userData.origRot || 0;
      jawBoneRef.current.rotation.x = base - applied * 0.35;
    }
  });

  return (
    <group ref={group} scale={[modelScale, modelScale, modelScale]} position={[0, 0, 0]}>
      <primitive object={scene} dispose={null} />
    </group>
  );
}

/* ---------- background ---------- */
function FullBackground({ url }) {
  const tex = useTexture(url);
  return (
    <mesh position={[0, 0, -4]} scale={[24, 14, 1]}>
      <planeGeometry args={[1, 1]} />
      <meshBasicMaterial map={tex} toneMapped={false} />
    </mesh>
  );
}

/* ---------- full component with safe audio handling ---------- */
export default function FullScreenAvatarProp({
  avatarUrl = "https://models.readyplayer.me/691079f1d28f9f8e50c6e261.glb",
  bgImage = "/model/blurred-image-of-an-office-space.webp",
  modelScale = 5.0,
  modelYOffset = -4.1,
  faceTilt = -0.6,
  audioUrl = null,
}) {
  const [analyser, setAnalyser] = useState(null);
  const audioCtxRef = useRef(null);
  const srcNodeRef = useRef(null);
  const audioElRef = useRef(null);

  // cleanup
  const disableAudio = () => {
    try {
      if (srcNodeRef.current?.disconnect) { try { srcNodeRef.current.disconnect(); } catch (e) {} srcNodeRef.current = null; }
      if (audioElRef.current) { try { audioElRef.current.pause(); } catch (e) {} audioElRef.current = null; }
      if (audioCtxRef.current) { try { audioCtxRef.current.close(); } catch (e) {} audioCtxRef.current = null; }
      setAnalyser(null);
    } catch (e) {}
  };

  // mic
  const enableMic = async () => {
    try {
      disableAudio();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      audioCtxRef.current = ctx;
      const srcNode = ctx.createMediaStreamSource(stream);
      srcNodeRef.current = srcNode;
      const a = ctx.createAnalyser();
      a.fftSize = 2048;
      srcNode.connect(a);
      setAnalyser(a);
    } catch (err) {
      console.error("enableMic error:", err);
    }
  };

  // file playback — always create fresh audio element
  const enableFilePlayback = async (url) => {
    try {
      if (!url) return;
      if (srcNodeRef.current?.disconnect) { try { srcNodeRef.current.disconnect(); } catch (e) {} srcNodeRef.current = null; }
      if (audioElRef.current) { try { audioElRef.current.pause(); } catch (e) {} audioElRef.current = null; }

      const audio = document.createElement("audio");
      audio.crossOrigin = "anonymous";
      audio.preload = "auto";
      audio.src = url;
      audio.controls = false;
      audioElRef.current = audio;

      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      audioCtxRef.current = ctx;

      const srcNode = ctx.createMediaElementSource(audio);
      srcNodeRef.current = srcNode;

      const a = ctx.createAnalyser();
      a.fftSize = 2048;
      srcNode.connect(a);
      srcNode.connect(ctx.destination);
      setAnalyser(a);

      try {
        await audio.play();
      } catch (playErr) {
        console.warn("audio play blocked, attempting ctx.resume():", playErr);
        try {
          await ctx.resume();
          await audio.play().catch((e) => console.warn("play still blocked:", e));
        } catch (resumeErr) {
          console.warn("resume failed:", resumeErr);
        }
      }
      console.log("Playing audio:", url);
    } catch (err) {
      console.error("enableFilePlayback error:", err);
      setAnalyser(null);
    }
  };

  // watch audioUrl prop
  useEffect(() => {
    if (audioUrl) enableFilePlayback(audioUrl);
    else disableAudio();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioUrl]);

  useEffect(() => () => disableAudio(), []);

  return (
    <div className="avatar-screen" style={{ position: "relative" }}>
      <div style={{ position: "absolute", zIndex: 10, left: 12, top: 12 }}>
        <button onClick={enableMic} style={{ marginRight: 8 }}>Enable Mic</button>
        <button onClick={() => enableFilePlayback("/sample-audio.wav")} style={{ marginRight: 8 }}>Play sample audio</button>
        <button onClick={disableAudio}>Disable Audio</button>
      </div>

      <Canvas camera={{ position: [0, 1.45, 1.55], fov: 28 }} gl={{ antialias: true }}>
        <ambientLight intensity={1.0} />
        <directionalLight position={[0, 2, 3]} intensity={1.1} />
        <pointLight position={[1.8, 0.8, 1.2]} intensity={0.6} color={"#a0c8ff"} />

        <Suspense fallback={null}>
          <FullBackground url={bgImage} />
          <AvatarHead url={avatarUrl} analyser={analyser} mouthOpen={0} modelScale={modelScale} modelYOffset={modelYOffset} faceTilt={faceTilt} />
        </Suspense>
      </Canvas>
    </div>
  );
}

