import glob
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np


# ========= 設定 =========
IMAGEDIR = Path("data/chess_captured")

# 「内側コーナー数」（横, 縦）※あなたの設定
CHECKERBOARD: Tuple[int, int] = (10, 7)

# 1マスのサイズ（mm）
SQUARE_SIZE: float = 19.0

# 失敗が多い場合は SB を試す
USE_SB: bool = True
# ========================


def calibrate_from_dir(image_dir: Path) -> tuple[float, np.ndarray, np.ndarray, float]:
    paths = sorted(glob.glob(str(image_dir / "chess_*.png")))
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    # 3D点（Z=0平面）
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    img_size = None
    used = 0

    for f in paths:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[skip] could not read: {f}")
            continue

        if img_size is None:
            img_size = (img.shape[1], img.shape[0])  # (w, h)

        if USE_SB and hasattr(cv2, "findChessboardCornersSB"):
            found, corners = cv2.findChessboardCornersSB(img, CHECKERBOARD)
            if not found:
                print(f"[skip] corners not found: {Path(f).name}")
                continue
            corners2 = corners.astype(np.float32)
        else:
            found, corners = cv2.findChessboardCorners(img, CHECKERBOARD, flags)
            if not found:
                print(f"[skip] corners not found: {Path(f).name}")
                continue
            corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners2)
        used += 1

    print(f"Detected corners in {used}/{len(paths)} images.")
    if used < 8:
        raise RuntimeError(
            "有効画像が少なすぎます（目安8枚以上、できれば15枚以上）。"
            " チェッカーボードの構造（外周が欠けていないか）と撮影条件を見直してください。"
        )

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # 平均再投影誤差
    total_err2 = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2)
        n = len(objpoints[i])
        total_err2 += err * err
        total_points += n
    mean_reproj_err = float(np.sqrt(total_err2 / total_points))

    return float(rms), K, dist, mean_reproj_err


def save_results(out_dir: Path, K: np.ndarray, dist: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "camera_calib.npz"
    np.savez(
        str(npz_path),
        K=K,
        dist=dist,
        checkerboard=np.array(CHECKERBOARD, dtype=np.int32),
        square_size=float(SQUARE_SIZE),
    )
    print(f"Saved: {npz_path.name}")

    yml_path = out_dir / "camera_calib.yaml"
    fs = cv2.FileStorage(str(yml_path), cv2.FILE_STORAGE_WRITE)
    fs.write("K", K)
    fs.write("dist", dist)
    fs.write("checkerboard_cols_rows", np.array(CHECKERBOARD, dtype=np.int32))
    fs.write("square_size", float(SQUARE_SIZE))
    fs.release()
    print(f"Saved: {yml_path.name}")


def main() -> None:
    print(f"Input dir: {IMAGEDIR.resolve()}")
    rms, K, dist, mean_err = calibrate_from_dir(IMAGEDIR)

    print("\n=== Result ===")
    print(f"RMS (OpenCV)           : {rms:.6f}")
    print(f"Mean reproj error (px) : {mean_err:.6f}")
    print("\nCamera Matrix K:\n", K)
    print("\nDistortion coeffs:\n", dist.ravel())

    save_results(IMAGEDIR, K, dist)
    print("\nDone.")


if __name__ == "__main__":
    main()
