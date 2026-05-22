/**
 * Image with a draggable circular marker overlay. Used on the AI
 * tracer page so the operator can hand-correct the AI's ball-at-
 * rest pick AND each per-frame ball-track position by dragging the
 * circles, instead of typing pixel coords into number inputs.
 *
 * Coordinates are exposed in NATIVE source-image pixels — the caller
 * doesn't need to think about the displayed image size. The component
 * scales drag deltas from displayed-pixel-space back to native using
 * the image's naturalWidth / clientWidth ratio.
 *
 * Usage:
 *   <DraggableMarker
 *     imageUrl="/uploads/clips/foo.jpg"
 *     x={500} y={300}      // native pixel coords; null/undefined = no marker rendered
 *     color="#0f0"
 *     onChange={(nx, ny) => ...}    // fired during drag (every move)
 *     onClick={(nx, ny) => ...}     // fired on a non-drag click (e.g. to place
 *                                    // a marker where AI missed entirely)
 *     label="rest"          // optional small label next to the marker
 *     style={...}           // forwarded to the wrapper div
 *   />
 */
import { useRef, useState } from "react";

export default function DraggableMarker({
  imageUrl,
  x,
  y,
  color = "#22dd66",
  size = 22,
  onChange,
  onClick,
  label,
  style,
  alt,
  imageStyle,
  disabled = false,
}) {
  const imgRef = useRef(null);
  const [drag, setDrag] = useState(null); // {pointerId, startX, startY, origX, origY, moved}
  const [, setTick] = useState(0); // force re-render after image loads (so we can read naturalWidth)

  function nativeToDisplay(nx, ny) {
    const img = imgRef.current;
    if (!img || !img.naturalWidth) return { dispX: nx, dispY: ny };
    const sx = img.clientWidth / img.naturalWidth;
    const sy = img.clientHeight / img.naturalHeight;
    return { dispX: nx * sx, dispY: ny * sy };
  }

  function displayToNative(dispX, dispY) {
    const img = imgRef.current;
    if (!img || !img.naturalWidth || !img.clientWidth) {
      return { nx: Math.round(dispX), ny: Math.round(dispY) };
    }
    const sx = img.naturalWidth / img.clientWidth;
    const sy = img.naturalHeight / img.clientHeight;
    return { nx: Math.round(dispX * sx), ny: Math.round(dispY * sy) };
  }

  function handlePointerDown(e) {
    if (disabled) return;
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.setPointerCapture?.(e.pointerId);
    setDrag({
      pointerId: e.pointerId,
      startX: e.clientX,
      startY: e.clientY,
      origX: x ?? 0,
      origY: y ?? 0,
      moved: false,
    });
  }

  function handlePointerMove(e) {
    if (!drag || drag.pointerId !== e.pointerId) return;
    const img = imgRef.current;
    if (!img || !img.clientWidth) return;
    const sx = (img.naturalWidth || img.clientWidth) / img.clientWidth;
    const sy = (img.naturalHeight || img.clientHeight) / img.clientHeight;
    const deltaNx = (e.clientX - drag.startX) * sx;
    const deltaNy = (e.clientY - drag.startY) * sy;
    if (
      !drag.moved
      && (Math.abs(e.clientX - drag.startX) > 2 || Math.abs(e.clientY - drag.startY) > 2)
    ) {
      drag.moved = true;
    }
    onChange?.(
      Math.max(0, Math.round(drag.origX + deltaNx)),
      Math.max(0, Math.round(drag.origY + deltaNy)),
    );
  }

  function handlePointerUp(e) {
    if (!drag) return;
    e.currentTarget.releasePointerCapture?.(e.pointerId);
    setDrag(null);
  }

  function handleImageClick(e) {
    if (disabled) return;
    if (!onClick) return;
    if (drag?.moved) return; // don't fire click on drag end
    const img = imgRef.current;
    if (!img) return;
    const rect = img.getBoundingClientRect();
    const dispX = e.clientX - rect.left;
    const dispY = e.clientY - rect.top;
    const { nx, ny } = displayToNative(dispX, dispY);
    onClick(nx, ny);
  }

  const showMarker = x != null && y != null && Number.isFinite(x) && Number.isFinite(y);
  const { dispX, dispY } = showMarker ? nativeToDisplay(x, y) : { dispX: 0, dispY: 0 };

  return (
    <div
      style={{
        position: "relative",
        display: "block",
        userSelect: "none",
        ...(style || {}),
      }}
    >
      <img
        ref={imgRef}
        src={imageUrl}
        alt={alt || ""}
        onClick={handleImageClick}
        onLoad={() => setTick((t) => t + 1)}
        style={{
          width: "100%",
          display: "block",
          background: "#000",
          cursor: onClick && !disabled ? "crosshair" : "default",
          ...(imageStyle || {}),
        }}
      />
      {showMarker && (
        <>
          <div
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
            style={{
              position: "absolute",
              left: dispX - size / 2,
              top: dispY - size / 2,
              width: size,
              height: size,
              borderRadius: "50%",
              border: `3px solid ${color}`,
              background: "rgba(255,255,255,0.15)",
              boxShadow: "0 0 0 1px rgba(0,0,0,0.6)",
              cursor: drag ? "grabbing" : "grab",
              touchAction: "none",
              pointerEvents: disabled ? "none" : "auto",
            }}
          />
          {label && (
            <div
              style={{
                position: "absolute",
                left: dispX + size / 2 + 4,
                top: dispY - size / 2 - 4,
                color,
                fontSize: 11,
                fontWeight: 600,
                textShadow: "0 0 3px black, 0 0 3px black",
                pointerEvents: "none",
              }}
            >
              {label}
            </div>
          )}
        </>
      )}
    </div>
  );
}
