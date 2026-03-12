import cv2
import numpy as np
import json
from pathlib import Path

from edsdk.camera_controller import CameraController


# ========= 設定 =========
CALIB_DIR = Path("data/chess_captured")  # cam_calib.py の出力先
NPZ_PATH = CALIB_DIR / "camera_calib.npz"
YAML_PATH = CALIB_DIR / "camera_calib.yaml"

OUT_DIR = Path("pose_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON_PATH = OUT_DIR / "pose_unity.json"

# Canon撮影設定（必要なら変更）
AV = 8
TV = 1 / 10
ISO = 160
IMAGE_QUALITY = "LJF"

# 推定方式：RANSACを使うなら True（推奨）
USE_RANSAC = False
RANSAC_REPROJ_ERR_PX = 5.0
RANSAC_CONFIDENCE = 0.999
RANSAC_ITERS = 200

# solvePnP / solvePnPRansac の flags
SOLVEPNP_FLAG = cv2.SOLVEPNP_ITERATIVE
# ========================


def load_intrinsics() -> tuple[np.ndarray, np.ndarray]:
    """cam_calib.py の出力に合わせて K, dist を読み込む（NPZ優先、YAMLはfallback）。"""
    if NPZ_PATH.exists():
        data = np.load(str(NPZ_PATH))
        K = data["K"].astype(np.float64)
        dist = data["dist"].astype(np.float64).reshape(-1, 1)
        return K, dist

    if YAML_PATH.exists():
        fs = cv2.FileStorage(str(YAML_PATH), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise RuntimeError(f"Failed to open: {YAML_PATH.resolve()}")
        K = fs.getNode("K").mat()
        dist = fs.getNode("dist").mat()
        fs.release()

        if K is None or K.size == 0:
            raise ValueError("Key 'K' not found in camera_calib.yaml")
        if dist is None or dist.size == 0:
            dist = np.zeros((5, 1), dtype=np.float64)

        return K.astype(np.float64), dist.astype(np.float64).reshape(-1, 1)

    raise FileNotFoundError(
        "Calibration file not found. Run cam_calib.py first.\n"
        f"Expected:\n  {NPZ_PATH.resolve()}\n  {YAML_PATH.resolve()}"
    )


def capture_one_image_from_canon() -> np.ndarray:
    """
    LiveViewなしで Canon から 1 枚撮影し、BGR画像（OpenCV表示用）を返す。
    camera.capture_numpy()[0] はRGB想定なので BGR に変換する。
    """
    with CameraController(register_property_events=False) as camera:
        camera.set_properties(av=AV, tv=TV, iso=ISO, image_quality=IMAGE_QUALITY)
        rgb = camera.capture_numpy()[0]
        if rgb.ndim == 2:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr


def collect_click_points_fullscreen(img_bgr: np.ndarray, labels: list[str]) -> np.ndarray:
    """
    画像をフルスクリーン表示して、len(labels) 個クリックして (u,v) を返す。
    操作:
      Left click: add | Backspace: undo | Enter: finish | Esc: quit
    表示:
      次にクリックすべきラベルを画面に表示する（A→B→...）
    """
    base = img_bgr.copy()
    vis = img_bgr.copy()
    clicked: list[tuple[int, int]] = []

    def redraw() -> None:
        nonlocal vis
        vis = base.copy()
        for i, (u, v) in enumerate(clicked, start=1):
            cv2.circle(vis, (u, v), 5, (0, 0, 255), -1)
            cv2.putText(
                vis, labels[i - 1], (u + 8, v - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
            )

    def draw_overlay(im: np.ndarray) -> np.ndarray:
        out = im.copy()
        n = len(labels)
        next_label = labels[len(clicked)] if len(clicked) < n else "(done)"
        cv2.putText(out,
                    "Left click: add | Backspace: undo | Enter: finish | Esc: quit",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(out, f"Clicked: {len(clicked)}/{n}   Next: {next_label}",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        return out

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicked) < len(labels):
                clicked.append((int(x), int(y)))
                redraw()

    win = "Click points (Fullscreen)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, draw_overlay(vis))
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(win)
            raise SystemExit("Cancelled by user.")

        if key == 8:  # Backspace
            if clicked:
                clicked.pop()
                redraw()

        if key == 13:  # Enter
            if len(clicked) == len(labels):
                break
            print(f"Need {len(labels)} points, but got {len(clicked)}")

    cv2.destroyWindow(win)
    return np.array(clicked, dtype=np.float32)


def rotation_matrix_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    """
    3x3回転行列 -> クォータニオン [x, y, z, w]
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    q = np.array([x, y, z, w], dtype=np.float64)
    q = q / np.linalg.norm(q)
    return q


def main() -> None:
    # 1) 内パラ読み込み
    K, dist = load_intrinsics()
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # 2) Canonで1枚撮影（ライブビューなし）
    img_bgr = capture_one_image_from_canon()

    # 3) 画像サイズを撮影画像から取得（要求仕様）
    # OpenCV: shape = (height, width, channels)
    height = int(img_bgr.shape[0])
    width = int(img_bgr.shape[1])
    print(f"Captured resolution: {width} x {height}")

    # 4) 既知3D点（mm）
    labels = list("ABCDEFGHIJKLMN")
    object_points_mm = np.array([
        # A
        [0.0,        -0.0,        -0.0],
        # B
        [-24.0372,    60.2031,    17.1096],
        # C
        [-24.3812,   118.816,    23.5571],
        # D
        [-34.8,      228.706,    29.1758],
        # E
        [95.4303,    46.8298,    40.6512],
        # F
        [81.4433,   289.803,    61.7818],
        # G
        [-504.171,    79.8121,   34.5012],
        # H
        [-724.94,    107.781,    35.64],
        # I
        [-675.507,   246.689,    31.8095],
        # J
        [-766.154,   251.885,     7.5298],
        # K
        [-874.557,   222.66,      1.79518],
        # L
        [-916.139,   254.286,     5.71086],
        # M
        [-910.412,    88.6131,   -2.51968],
        # N
        [-816.936,    26.4739,    8.34665],
    ], dtype=np.float32)

    # 5) クリックで2D点取得（A→N）
    image_points = collect_click_points_fullscreen(img_bgr, labels)

    # 6) 外部推定（RANSAC有無）
    if USE_RANSAC:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points_mm, image_points, K, dist,
            flags=SOLVEPNP_FLAG,
            reprojectionError=RANSAC_REPROJ_ERR_PX,
            confidence=RANSAC_CONFIDENCE,
            iterationsCount=RANSAC_ITERS
        )
        if (not success) or (inliers is None) or (len(inliers) < 4):
            raise RuntimeError("solvePnPRansac failed or not enough inliers.")
        inlier_idx = inliers.reshape(-1).tolist()
    else:
        success, rvec, tvec = cv2.solvePnP(
            object_points_mm, image_points, K, dist,
            flags=SOLVEPNP_FLAG
        )
        if not success:
            raise RuntimeError("solvePnP failed.")
        inlier_idx = list(range(len(object_points_mm)))  # 全点扱い

    # solvePnP / solvePnPRansac の出力:
    # rvec, tvec が得られている前提

    R_w2c, _ = cv2.Rodrigues(rvec)
    t_w2c_mm = tvec.astype(np.float64).reshape(3, 1)  # (3,1)

    # ---- 1) OpenCV世界で camera-to-world とカメラ中心を計算 ----
    R_c2w = R_w2c.T
    C_mm = -R_c2w @ t_w2c_mm  # (3,1)  world座標系でのカメラ位置（mm）

    # ---- 2) Unity用に変換（あなたの例と同じ：Y反転 + /1000） ----
    # 位置（mm -> m, Y反転）
    C_unity = np.array([C_mm[0, 0] / 1000.0,
                        -C_mm[1, 0] / 1000.0,
                        C_mm[2, 0] / 1000.0], dtype=np.float64)

    # 回転（Y反転を両側から：S R S）
    S = np.diag([1.0, -1.0, 1.0]).astype(np.float64)
    R_unity = S @ R_c2w @ S

    # クォータニオン（x,y,z,w）
    q_unity_xyzw = rotation_matrix_to_quaternion_xyzw(R_unity)

    print("Unity position (m):", C_unity)
    print("Unity quaternion [x,y,z,w]:", q_unity_xyzw)


    # 8) 再投影誤差（参考表示）
    proj, _ = cv2.projectPoints(object_points_mm, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - image_points, axis=1)
    err_mean_all = float(np.mean(err))
    err_max_all = float(np.max(err))

    if USE_RANSAC:
        mask = np.zeros((len(object_points_mm),), dtype=bool)
        mask[np.array(inlier_idx, dtype=int)] = True
        err_mean_inliers = float(np.mean(err[mask]))
    else:
        err_mean_inliers = err_mean_all

    print("Reprojection error mean (all) [px]:", err_mean_all)
    print("Reprojection error mean (inliers) [px]:", err_mean_inliers)

    # 9) Unityに読み込みやすいJSONを書き出し
    # distはUnity標準カメラには直接使わないが、記録用に入れる
    payload = {
        "image_width": width,
        "image_height": height,

        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "dist": dist.reshape(-1).astype(float).tolist(),

        # 外パラ（Unity用）
        "camera_position_m": C_unity.astype(float).tolist(),
        "camera_rotation_xyzw": q_unity_xyzw.astype(float).tolist(),

        # デバッグ/検証用（任意）
        "use_ransac": bool(USE_RANSAC),
        "inlier_indices": inlier_idx,
        "reprojection_error_mean_all_px": err_mean_all,
        "reprojection_error_mean_inliers_px": err_mean_inliers,
        "reprojection_error_max_all_px": err_max_all,
    }

    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved Unity-friendly JSON: {OUT_JSON_PATH.resolve()}")

    # 10) 可視化（クリック点=赤、再投影点=青リング）
    vis = img_bgr.copy()
    for i, ((u, v), (pu, pv)) in enumerate(zip(image_points, proj), start=1):
        u_i, v_i = int(round(u)), int(round(v))
        pu_i, pv_i = int(round(pu)), int(round(pv))
        cv2.circle(vis, (u_i, v_i), 6, (0, 0, 255), -1)       # clicked
        cv2.circle(vis, (pu_i, pv_i), 6, (255, 0, 0), 2)      # projected
        cv2.putText(vis, labels[i - 1], (u_i + 6, v_i - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(vis, "Clicked=red | Projected=blue ring (press any key to close)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.namedWindow("Pose Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Pose Result", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
