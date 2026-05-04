import { Icon } from "./Brand.jsx";
import SelfieCamera from "./SelfieCamera.jsx";

/**
 * One reusable selfie slot used by Register.jsx for the lead golfer and each
 * group member. The parent owns `openCameraId` so only one device camera is
 * mounted at a time across all SelfieField instances.
 */
export default function SelfieField({
  id,
  openCameraId,
  setOpenCameraId,
  preview,
  onPicked,
  size = "lg",
}) {
  const open = openCameraId === id;
  const buttonLabel = preview ? "Retake outfit photo" : "Open camera";

  if (open) {
    return (
      <SelfieCamera
        onCapture={(file) => { onPicked(file); setOpenCameraId(null); }}
        onCancel={() => setOpenCameraId(null)}
      />
    );
  }
  if (preview) {
    return (
      <div className="selfie-preview">
        <img src={preview} alt="outfit" style={size === "sm" ? { width: 56, height: 76 } : undefined} />
        <div style={{ flex: 1 }}>
          <div className="ok-text inline"><Icon name="check" size={16} /> Got it</div>
          <div className="small muted">We'll match clips to this outfit.</div>
        </div>
        <button type="button" className="secondary small" onClick={() => setOpenCameraId(id)}>
          <Icon name="retake" size={14} /> Retake
        </button>
      </div>
    );
  }
  return (
    <button type="button" onClick={() => setOpenCameraId(id)}>
      <Icon name="camera" size={16} /> {buttonLabel}
    </button>
  );
}
