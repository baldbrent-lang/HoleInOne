import { useEffect, useRef, useState } from "react";

/**
 * In-page camera capture optimised for full-outfit golfer selfies.
 *
 * Defaults to the rear camera (wider lens, higher res than the front cam on
 * most phones), shows a faint head-to-toe silhouette guide, and returns a
 * JPEG File via onCapture. Falls back to a file input if getUserMedia is
 * unavailable or permission is denied.
 */
export default function SelfieCamera({ onCapture, onCancel }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const fileFallbackRef = useRef(null);
  const [error, setError] = useState(null);
  const [ready, setReady] = useState(false);
  const [facing, setFacing] = useState("environment"); // "environment" | "user"

  useEffect(() => {
    let cancelled = false;

    async function start() {
      setReady(false);
      setError(null);
      stopStream();

      if (!navigator.mediaDevices?.getUserMedia) {
        setError("This browser doesn't support in-page camera. Use the file picker below.");
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: { ideal: facing },
            width: { ideal: 1080 },
            height: { ideal: 1920 },
          },
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        const video = videoRef.current;
        if (video) {
          video.srcObject = stream;
          // iOS requires playsInline + a user gesture; we start from the
          // parent's onClick so this play() call is allowed.
          await video.play().catch(() => {});
          setReady(true);
        }
      } catch (e) {
        setError(e?.message || "Couldn't open camera. Use the file picker below.");
      }
    }

    start();
    return () => {
      cancelled = true;
      stopStream();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [facing]);

  function stopStream() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }

  function capture() {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    // Mirror for the user-facing camera so the captured image matches what the
    // golfer saw on screen (we mirrored the preview via CSS).
    if (facing === "user") {
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const file = new File([blob], "selfie.jpg", { type: "image/jpeg" });
        stopStream();
        onCapture(file);
      },
      "image/jpeg",
      0.9,
    );
  }

  function handleFallbackPick(file) {
    if (!file) return;
    onCapture(file);
  }

  return (
    <div className="camera-frame">
      <video
        ref={videoRef}
        playsInline
        muted
        autoPlay
        className={facing === "user" ? "mirror" : ""}
      />

      {/* Silhouette guide overlay */}
      <svg
        className="silhouette-guide"
        viewBox="0 0 100 200"
        preserveAspectRatio="xMidYMid meet"
        aria-hidden="true"
      >
        {/* Head */}
        <circle cx="50" cy="22" r="11" />
        {/* Neck + torso */}
        <path d="M44 33 L44 42 L36 50 L36 110 L44 110 L44 165 L46 195 L42 195" />
        <path d="M56 33 L56 42 L64 50 L64 110 L56 110 L56 165 L54 195 L58 195" />
        {/* Crotch */}
        <path d="M44 110 L50 118 L56 110" />
        {/* Arms */}
        <path d="M36 50 L26 100 L24 130" />
        <path d="M64 50 L74 100 L76 130" />
      </svg>

      <div className="camera-coach">Fit head to toe inside the outline</div>

      <div className="camera-controls">
        <button type="button" className="secondary" onClick={onCancel}>
          Cancel
        </button>
        <button
          type="button"
          onClick={() => setFacing(facing === "user" ? "environment" : "user")}
          className="secondary"
          title="Flip camera"
        >
          Flip
        </button>
        <button type="button" disabled={!ready} onClick={capture}>
          {ready ? "Capture" : "Loading…"}
        </button>
      </div>

      {error && (
        <div className="camera-error">
          <p className="small">{error}</p>
          <input
            ref={fileFallbackRef}
            type="file"
            accept="image/*"
            capture="environment"
            style={{ display: "none" }}
            onChange={(e) => handleFallbackPick(e.target.files?.[0])}
          />
          <button type="button" onClick={() => fileFallbackRef.current?.click()}>
            Use file picker
          </button>
        </div>
      )}
    </div>
  );
}
