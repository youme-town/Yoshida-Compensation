import time
from pathlib import Path

import cv2
import numpy as np

from edsdk.camera_controller import CameraController


# ========= 設定 =========
CAPTUREDIR = Path("data/chess_captured")

NUM_IMAGES = 20
INTERVAL_SEC = 5.0

# 表示負荷軽減（表示だけ縮小）
PREVIEW_SCALE = 0.6
SHOW_OVERLAY = True

# Canon露出（必要に応じて調整）
AV = 5
TV = 1 / 15
ISO = 200
IMAGE_QUALITY = "LJF"
# ========================


def _to_bgr_from_liveview(lv_rgb: np.ndarray) -> np.ndarray:
    # grab_live_view_numpy() は通常 RGB
    if lv_rgb.ndim == 2:
        return cv2.cvtColor(lv_rgb, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(lv_rgb, cv2.COLOR_RGB2BGR)


def _to_gray_from_capture(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def main() -> None:
    CAPTUREDIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving to: {CAPTUREDIR.resolve()}")
    print(f"Will capture {NUM_IMAGES} images every {INTERVAL_SEC} seconds.")
    print("Press 'q' to quit.")

    with CameraController(register_property_events=False) as camera:
        camera.set_properties(av=AV, tv=TV, iso=ISO, image_quality=IMAGE_QUALITY)

        # LiveView開始
        camera.start_live_view()

        # アスペクト比を崩したくないので AUTOSIZE 推奨
        cv2.namedWindow("Canon LiveView", cv2.WINDOW_AUTOSIZE)

        next_capture_t = time.time() + INTERVAL_SEC
        captured_count = 0

        try:
            while True:
                now = time.time()

                # ---- LiveView プレビュー ----
                try:
                    lv_rgb = camera.grab_live_view_numpy()
                    frame = _to_bgr_from_liveview(lv_rgb)

                    if PREVIEW_SCALE != 1.0:
                        frame = cv2.resize(
                            frame, None,
                            fx=PREVIEW_SCALE, fy=PREVIEW_SCALE,
                            interpolation=cv2.INTER_AREA
                        )

                    if SHOW_OVERLAY:
                        remain = max(0.0, next_capture_t - now)
                        cv2.putText(frame, f"Captured: {captured_count}/{NUM_IMAGES}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"Next capture in: {remain:0.1f}s   (q: quit)", (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.imshow("Canon LiveView", frame)

                except Exception as e:
                    print(f"[warn] liveview grab failed: {e}")

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Quit requested.")
                    break

                # ---- 定刻で保存用撮影（高解像度） ----
                if now >= next_capture_t and captured_count < NUM_IMAGES:
                    print(f"[{captured_count+1}/{NUM_IMAGES}] Capturing & saving...")

                    rgb = camera.capture_numpy()[0]
                    gray = _to_gray_from_capture(rgb)

                    out_path = CAPTUREDIR / f"chess_{captured_count:02d}.png"
                    if cv2.imwrite(str(out_path), gray):
                        print(f"Saved: {out_path.name}")
                        captured_count += 1
                    else:
                        print(f"[warn] failed to save: {out_path}")

                    next_capture_t = time.time() + INTERVAL_SEC

                    if captured_count >= NUM_IMAGES:
                        print("All images captured.")
                        break

                time.sleep(0.002)

        finally:
            try:
                camera.stop_live_view()
            except Exception:
                pass
            cv2.destroyAllWindows()

    print("Done.")


if __name__ == "__main__":
    main()
