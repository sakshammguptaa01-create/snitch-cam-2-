import { useCallback, useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import { filterPhones } from "@/lib/phone-filter";
import { detectMultiScale } from "@/lib/detect";

// Each violation saved to the log gets an id, timestamp, confidence score and optional duration.
type LogEntry = { id: number; time: string; score: number; duration?: number };

export default function SnitchCam() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const rafRef = useRef<number | null>(null);
  const audioRef = useRef<AudioContext | null>(null);
  const lastBeepRef = useRef(0);
  const violationStartRef = useRef<number | null>(null);
  const logIdRef = useRef(0);
  const demoImageRef = useRef<string | null>(null);

  const [status, setStatus] = useState<"idle" | "loading" | "running" | "error">("idle");
  const [message, setMessage] = useState("Model not loaded");
  const [detected, setDetected] = useState(false);
  const [score, setScore] = useState(0);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [soundOn, setSoundOn] = useState(true);
  const [notifyOn, setNotifyOn] = useState(false);
  const [fps, setFps] = useState(0);
  const [confidence, setConfidence] = useState(0.5);
  const [sensitivity, setSensitivity] = useState(6);
  // Strict mode adds shape/size/rival-object checks that reject look-alikes such as a computer mouse.
  const [strict, setStrict] = useState(true);
  // Long-range mode runs extra upscaled tile passes so small/distant phones are seen.
  const [longRange, setLongRange] = useState(true);

  // Refs mirror slider values so the detection loop reads the latest setting without re-rendering.
  const confidenceRef = useRef(confidence);
  const sensitivityRef = useRef(sensitivity);
  const strictRef = useRef(strict);
  const longRangeRef = useRef(longRange);
  // hitRef / missRef count consecutive frames with/without a phone to reduce flickering alerts.
  const hitRef = useRef(0);
  const missRef = useRef(0);
  useEffect(() => {
    confidenceRef.current = confidence;
  }, [confidence]);
  useEffect(() => {
    sensitivityRef.current = sensitivity;
  }, [sensitivity]);
  useEffect(() => {
    strictRef.current = strict;
  }, [strict]);
  useEffect(() => {
    longRangeRef.current = longRange;
  }, [longRange]);

  const beep = useCallback(() => {
    if (!soundOn) return;
    const now = Date.now();
    if (now - lastBeepRef.current < 900) return;
    lastBeepRef.current = now;
    try {
      const ctx =
        audioRef.current ??
        new (
          window.AudioContext ||
          (window as never as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
        )();
      audioRef.current = ctx;
      if (ctx.state === "suspended") void ctx.resume();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "square";
      osc.frequency.setValueAtTime(880, ctx.currentTime);
      osc.frequency.setValueAtTime(1320, ctx.currentTime + 0.12);
      gain.gain.setValueAtTime(0.0001, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.25, ctx.currentTime + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.35);
      osc.connect(gain).connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.36);
    } catch {
      /* audio unavailable */
    }
  }, [soundOn]);

  const notify = useCallback(() => {
    if (!notifyOn || typeof Notification === "undefined") return;
    if (Notification.permission !== "granted") return;
    try {
      new Notification("Snitch Cam Alert", {
        body: "Phone usage detected in frame.",
        tag: "snitch-cam",
      });
    } catch {
      /* notifications unavailable */
    }
  }, [notifyOn]);

  const requestNotify = async () => {
    if (typeof Notification === "undefined") return;
    const perm = await Notification.requestPermission();
    setNotifyOn(perm === "granted");
  };

  const stop = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    const stream = videoRef.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((t) => t.stop());
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.src = "";
      videoRef.current.pause();
      videoRef.current.style.opacity = "1";
    }
    demoImageRef.current = null;
    violationStartRef.current = null;
    setDetected(false);
    setScore(0);
    setFps(0);
    setStatus("idle");
    setMessage("Surveillance stopped");
  }, []);

  const loadModel = async () => {
    if (modelRef.current) return;
    setMessage("Loading pretrained detection model…");
    try {
      await tf.ready();
      modelRef.current = await cocoSsd.load({ base: "lite_mobilenet_v2" });
    } catch (e) {
      // If WebGL is unavailable (some headless/older machines), fall back to CPU.
      if (String(e).includes("WebGL")) {
        setMessage("WebGL unavailable, switching to CPU backend (slower)…");
        await tf.setBackend("cpu");
        await tf.ready();
        modelRef.current = await cocoSsd.load({ base: "lite_mobilenet_v2" });
      } else {
        throw e;
      }
    }
  };

  // Draw red bounding boxes and labels for every phone detection on the canvas overlay.
  const drawPredictions = (
    ctx: CanvasRenderingContext2D,
    phones: cocoSsd.DetectedObject[],
    width: number,
    height: number,
  ) => {
    ctx.clearRect(0, 0, width, height);
    ctx.lineWidth = 3;
    ctx.font = "600 16px ui-sans-serif, system-ui";
    for (const p of phones) {
      const [x, y, w, h] = p.bbox;
      ctx.strokeStyle = "rgb(255,80,80)";
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = "rgba(255,80,80,0.85)";
      ctx.fillRect(x, y - 22, ctx.measureText(p.class).width + 60, 22);
      ctx.fillStyle = "#fff";
      ctx.fillText(`${p.class} ${(p.score * 100).toFixed(0)}%`, x + 6, y - 6);
    }
  };

  const registerViolation = (best: cocoSsd.DetectedObject) => {
    if (violationStartRef.current === null) {
      violationStartRef.current = Date.now();
      notify();
      setLogs((l) =>
        [
          {
            id: ++logIdRef.current,
            time: new Date().toLocaleTimeString(),
            score: best.score,
          },
          ...l,
        ].slice(0, 40),
      );
    }
    setDetected(true);
    setScore(best.score);
    beep();
  };

  const clearViolation = () => {
    if (violationStartRef.current !== null) {
      const dur = (Date.now() - violationStartRef.current) / 1000;
      const id = logIdRef.current;
      violationStartRef.current = null;
      setLogs((l) => l.map((e) => (e.id === id ? { ...e, duration: dur } : e)));
    }
    setDetected(false);
    setScore(0);
  };

  // Decide whether a stable violation has started or ended based on consecutive frames.
  const evaluateFrame = (best: cocoSsd.DetectedObject | undefined) => {
    // Higher sensitivity slider => fewer consecutive frames needed to trigger.
    const needed = 11 - sensitivityRef.current;

    if (best) {
      hitRef.current++;
      missRef.current = 0;
    } else {
      missRef.current++;
      hitRef.current = 0;
    }

    if (best && hitRef.current >= needed) {
      registerViolation(best);
    } else if (!best && missRef.current >= needed) {
      clearViolation();
    }
  };

  const start = useCallback(async () => {
    try {
      setStatus("loading");
      setMessage("Requesting camera…");
      demoImageRef.current = null;
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.style.opacity = "1";
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      await loadModel();

      setStatus("running");
      setMessage("Monitoring live feed");

      let last = performance.now();
      let frames = 0;

      const loop = async () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const model = modelRef.current;
        if (!video || !canvas || !model || video.readyState < 2) {
          rafRef.current = requestAnimationFrame(() => void loop());
          return;
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        const preds = await detectMultiScale(model, video, {
          width: canvas.width,
          height: canvas.height,
          longRange: longRangeRef.current,
        });

        const phones = filterPhones(preds, {
          threshold: confidenceRef.current,
          strict: strictRef.current,
          frameWidth: canvas.width,
          frameHeight: canvas.height,
          longRange: longRangeRef.current,
        });
        const best = phones.sort((a, b) => b.score - a.score)[0];

        if (ctx) drawPredictions(ctx, phones, canvas.width, canvas.height);
        evaluateFrame(best);

        frames++;
        const now = performance.now();
        if (now - last > 1000) {
          setFps(Math.round((frames * 1000) / (now - last)));
          frames = 0;
          last = now;
        }

        rafRef.current = requestAnimationFrame(() => void loop());
      };

      rafRef.current = requestAnimationFrame(() => void loop());
    } catch (e) {
      setStatus("error");
      setMessage(e instanceof Error ? e.message : "Could not start camera");
    }
  }, [beep, notify]);

  const runDemo = useCallback(
    async (src: string, label: string) => {
      try {
        setStatus("loading");
        setMessage(`Loading demo image: ${label}…`);
        await loadModel();

        const img = new Image();
        img.crossOrigin = "anonymous";
        img.src = src;
        await new Promise<void>((resolve, reject) => {
          img.onload = () => resolve();
          img.onerror = () => reject(new Error("Failed to load demo image"));
        });
        imageRef.current = img;
        demoImageRef.current = src;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) return;

        // Stop any live camera stream so the demo image is visible
        const stream = video.srcObject as MediaStream | null;
        stream?.getTracks().forEach((t) => t.stop());
        video.srcObject = null;
        video.style.opacity = "0";

        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        }

        const preds = await detectMultiScale(modelRef.current!, img, {
          width: canvas.width,
          height: canvas.height,
          longRange: longRangeRef.current,
        });
        const phones = filterPhones(preds, {
          threshold: confidenceRef.current,
          strict: strictRef.current,
          frameWidth: canvas.width,
          frameHeight: canvas.height,
          longRange: longRangeRef.current,
        });
        const best = phones.sort((a, b) => b.score - a.score)[0];

        if (ctx) drawPredictions(ctx, phones, canvas.width, canvas.height);

        hitRef.current = 0;
        missRef.current = 0;
        violationStartRef.current = null;
        evaluateFrame(best);

        setStatus("idle");
        setMessage(
          best
            ? `Demo: phone detected at ${(best.score * 100).toFixed(0)}% confidence`
            : "Demo: no phone detected",
        );
      } catch (e) {
        setStatus("error");
        setMessage(e instanceof Error ? e.message : "Demo failed");
      }
    },
    [beep, notify],
  );

  // Play a prerecorded sample video through the same detector loop.
  const runDemoVideo = useCallback(async () => {
    try {
      setStatus("loading");
      setMessage("Loading sample video…");
      await loadModel();

      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;

      // Stop any active camera stream before playing the file.
      const stream = video.srcObject as MediaStream | null;
      stream?.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
      video.src = "/demo/sample-video.webm";
      video.loop = true;
      video.muted = true;
      video.style.opacity = "1";
      demoImageRef.current = "/demo/sample-video.mp4";

      await video.play();
      setStatus("running");
      setMessage("Playing sample video");

      let last = performance.now();
      let frames = 0;

      const loop = async () => {
        const v = videoRef.current;
        const c = canvasRef.current;
        const model = modelRef.current;
        if (!v || !c || !model || v.paused || v.ended) {
          rafRef.current = requestAnimationFrame(() => void loop());
          return;
        }

        c.width = v.videoWidth;
        c.height = v.videoHeight;
        const ctx = c.getContext("2d");
        const preds = await detectMultiScale(model, v, {
          width: c.width,
          height: c.height,
          longRange: longRangeRef.current,
        });

        const phones = filterPhones(preds, {
          threshold: confidenceRef.current,
          strict: strictRef.current,
          frameWidth: c.width,
          frameHeight: c.height,
          longRange: longRangeRef.current,
        });
        const best = phones.sort((a, b) => b.score - a.score)[0];

        if (ctx) drawPredictions(ctx, phones, c.width, c.height);
        evaluateFrame(best);

        frames++;
        const now = performance.now();
        if (now - last > 1000) {
          setFps(Math.round((frames * 1000) / (now - last)));
          frames = 0;
          last = now;
        }

        rafRef.current = requestAnimationFrame(() => void loop());
      };

      rafRef.current = requestAnimationFrame(() => void loop());
    } catch (e) {
      setStatus("error");
      setMessage(e instanceof Error ? e.message : "Sample video failed");
    }
  }, [beep, notify]);

  useEffect(() => () => stop(), [stop]);

  const violations = logs.length;

  return (
    <div className="grid gap-6 lg:grid-cols-[1.6fr_1fr]">
      <section
        className={`relative overflow-hidden rounded-2xl border-2 bg-card shadow-panel transition-colors ${
          detected ? "border-destructive animate-alert" : "border-border"
        }`}
      >
        <div className="flex items-center justify-between border-b border-border/70 px-4 py-3">
          <div className="flex items-center gap-2">
            <span
              className={`size-2.5 rounded-full ${
                status === "running" ? "bg-destructive animate-pulse" : "bg-muted-foreground"
              }`}
            />
            <span className="font-mono text-xs tracking-widest text-muted-foreground uppercase">
              Cam 01 · Live Feed
            </span>
          </div>
          <span className="font-mono text-xs text-muted-foreground">{fps} FPS</span>
        </div>

        <div className="relative aspect-video w-full bg-black">
          <video
            ref={videoRef}
            playsInline
            muted
            className="absolute inset-0 size-full object-cover"
          />
          <canvas ref={canvasRef} className="absolute inset-0 size-full object-cover" />
          <div className="pointer-events-none absolute inset-0 bg-scanlines opacity-30" />

          {status !== "running" && !demoImageRef.current && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/70 px-6 text-center">
              <p className="font-mono text-sm tracking-widest text-muted-foreground uppercase">
                {status === "loading" ? "Initialising" : status === "error" ? "Error" : "Standby"}
              </p>
              <p className="max-w-sm text-sm text-muted-foreground">{message}</p>
            </div>
          )}

          {detected && (
            <div className="absolute left-1/2 top-4 -translate-x-1/2 rounded-full bg-destructive px-4 py-1.5 font-mono text-xs font-bold tracking-widest text-destructive-foreground uppercase">
              ⚠ Phone Usage Detected
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-3 px-4 py-4">
          {status === "running" ? (
            <button onClick={stop} className="btn-danger">
              Stop Monitoring
            </button>
          ) : (
            <button
              onClick={() => void start()}
              disabled={status === "loading"}
              className="btn-primary"
            >
              {status === "loading" ? "Loading…" : "Start Monitoring"}
            </button>
          )}
          <button
            onClick={() => setSoundOn((s) => !s)}
            className={soundOn ? "btn-ghost-on" : "btn-ghost"}
          >
            {soundOn ? "🔊 Beep On" : "🔇 Beep Off"}
          </button>
          <button
            onClick={() => void requestNotify()}
            className={notifyOn ? "btn-ghost-on" : "btn-ghost"}
          >
            {notifyOn ? "🔔 Notifications On" : "🔕 Enable Notifications"}
          </button>
          <button
            onClick={() => setStrict((s) => !s)}
            className={strict ? "btn-ghost-on" : "btn-ghost"}
            title="Rejects look-alike objects such as a computer mouse, remote or book"
          >
            {strict ? "🎯 Strict Filter On" : "🎯 Strict Filter Off"}
          </button>
          <button
            onClick={() => setLongRange((v) => !v)}
            className={longRange ? "btn-ghost-on" : "btn-ghost"}
            title="Scans zoomed-in tiles of the frame so phones far from the camera are still detected"
          >
            {longRange ? "🔭 Long Range On" : "🔭 Long Range Off"}
          </button>
        </div>

        <div className="grid gap-4 border-t border-border/70 px-4 py-4 sm:grid-cols-2">
          <label className="block">
            <span className="flex items-center justify-between font-mono text-[11px] tracking-widest text-muted-foreground uppercase">
              Confidence Threshold
              <span className="text-foreground">{(confidence * 100).toFixed(0)}%</span>
            </span>
            <input
              type="range"
              min={20}
              max={90}
              step={1}
              value={Math.round(confidence * 100)}
              onChange={(e) => setConfidence(Number(e.target.value) / 100)}
              className="mt-2 w-full accent-[var(--color-destructive)]"
            />
            <span className="mt-1 block text-[11px] text-muted-foreground">
              Lower = catches more phones (more false alarms). Higher = only very sure detections.
            </span>
          </label>

          <label className="block">
            <span className="flex items-center justify-between font-mono text-[11px] tracking-widest text-muted-foreground uppercase">
              Detection Sensitivity
              <span className="text-foreground">{sensitivity}/10</span>
            </span>
            <input
              type="range"
              min={1}
              max={10}
              step={1}
              value={sensitivity}
              onChange={(e) => setSensitivity(Number(e.target.value))}
              className="mt-2 w-full accent-[var(--color-destructive)]"
            />
            <span className="mt-1 block text-[11px] text-muted-foreground">
              Needs {11 - sensitivity} consecutive frame{11 - sensitivity === 1 ? "" : "s"} before
              alerting — lower is calmer, higher reacts instantly.
            </span>
          </label>
        </div>

        <div className="border-t border-border/70 px-4 py-4">
          <p className="font-mono text-[11px] tracking-widest text-muted-foreground uppercase">
            Demo Mode (no camera needed)
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              onClick={() => void runDemo("/demo/with-phone.jpg", "Phone in hand")}
              disabled={status === "loading"}
              className="btn-ghost"
            >
              Test: Phone in Frame
            </button>
            <button
              onClick={() => void runDemo("/demo/no-phone.jpg", "No phone")}
              disabled={status === "loading"}
              className="btn-ghost"
            >
              Test: No Phone
            </button>
            <button
              onClick={() => void runDemoVideo()}
              disabled={status === "loading"}
              className="btn-ghost"
            >
              ▶ Run Sample Video
            </button>
          </div>
          <p className="mt-2 text-[11px] text-muted-foreground">
            Use these if the exam computer has no webcam or camera permission is blocked. The sample
            video loops between a phone-in-frame scene and a clear scene.
          </p>
        </div>
      </section>

      <aside className="flex flex-col gap-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-xl border border-border bg-card p-4">
            <p className="font-mono text-[11px] tracking-widest text-muted-foreground uppercase">
              Status
            </p>
            <p
              className={`mt-1 text-lg font-bold ${detected ? "text-destructive" : "text-accent"}`}
            >
              {detected ? "VIOLATION" : status === "running" ? "CLEAR" : "OFFLINE"}
            </p>
          </div>
          <div className="rounded-xl border border-border bg-card p-4">
            <p className="font-mono text-[11px] tracking-widest text-muted-foreground uppercase">
              Confidence
            </p>
            <p className="mt-1 text-lg font-bold text-foreground">{(score * 100).toFixed(0)}%</p>
          </div>
          <div className="col-span-2 rounded-xl border border-border bg-card p-4">
            <p className="font-mono text-[11px] tracking-widest text-muted-foreground uppercase">
              Total Violations
            </p>
            <p className="mt-1 text-3xl font-black text-destructive">{violations}</p>
          </div>
        </div>

        <div className="rounded-xl border border-border bg-card p-4">
          <p className="font-mono text-[11px] tracking-widest text-muted-foreground uppercase">
            Detection Tips
          </p>
          <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-muted-foreground">
            <li>Keep the phone clearly visible and facing the camera.</li>
            <li>Use bright, even lighting — shadows reduce accuracy.</li>
            <li>
              Phone far from the camera? Keep <span className="text-foreground">Long Range</span> on
              — it scans zoomed tiles of the frame so small, distant phones are still caught
              (slightly lower FPS).
            </li>
            <li>Plain back covers work better than heavily patterned ones.</li>
            <li>Avoid covering the phone with hands, books or clothing.</li>
            <li>
              Getting false alarms on a computer mouse or remote? Keep{" "}
              <span className="text-foreground">Strict Filter</span> on and raise the confidence
              threshold to 60–70%.
            </li>
          </ul>
        </div>

        <div className="flex min-h-64 flex-1 flex-col rounded-xl border border-border bg-card">
          <p className="border-b border-border/70 px-4 py-3 font-mono text-[11px] tracking-widest text-muted-foreground uppercase">
            Violation Log
          </p>
          <ul className="flex-1 overflow-y-auto p-2 font-mono text-xs">
            {logs.length === 0 && (
              <li className="p-3 text-muted-foreground">No violations recorded.</li>
            )}
            {logs.map((l) => (
              <li
                key={l.id}
                className="flex items-center justify-between rounded-lg px-3 py-2 odd:bg-secondary/40"
              >
                <span className="text-foreground">{l.time}</span>
                <span className="text-muted-foreground">
                  {(l.score * 100).toFixed(0)}%
                  {l.duration ? ` · ${l.duration.toFixed(1)}s` : " · ongoing"}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </aside>
    </div>
  );
}
