from pathlib import Path
import importlib.util
import sys
import types

import cv2
import numpy as np


def _install_optional_dependency_stubs() -> None:
    if "colour" not in sys.modules:
        colour = types.ModuleType("colour")
        colour.characterisation = types.SimpleNamespace(
            polynomial_expansion_Finlayson2015=lambda *args, **kwargs: None
        )
        colour.algebra = types.SimpleNamespace(vecmul=lambda *args, **kwargs: None)
        colour.XYZ_to_sRGB = lambda image, apply_cctf_encoding=False: image
        sys.modules["colour"] = colour

    if "psutil" not in sys.modules:
        psutil = types.ModuleType("psutil")
        psutil.virtual_memory = lambda: types.SimpleNamespace(available=1 << 30)
        sys.modules["psutil"] = psutil


def _load_sample_module():
    _install_optional_dependency_stubs()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sample_path = repo_root / "examples" / "python" / "sample.py"
    spec = importlib.util.spec_from_file_location("sample_module", sample_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {sample_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sample = _load_sample_module()


def _save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _save_identity_p2c_map(path: Path, width: int, height: int) -> None:
    rows = []
    for y in range(height):
        for x in range(width):
            rows.append([float(x), float(y), float(x), float(y)])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(rows, dtype=np.float32))


def test_sample_runs_with_mocked_camera_and_projector(tmp_path, monkeypatch) -> None:
    width, height = 4, 3
    data_dir = tmp_path / "data"
    target_dir = data_dir / "target_images"
    map_path = tmp_path / "maps" / "p2c.npy"

    _save_identity_p2c_map(map_path, width, height)

    target_image = np.zeros((height, width, 3), dtype=np.uint8)
    target_image[..., 0] = 32
    target_image[..., 1] = 128
    target_image[..., 2] = 224
    _save_rgb_image(target_dir / "target.png", target_image)

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[projector]
gamma = 2.2
width = {width}
height = {height}
pos_x = 0
pos_y = 0

[camera]
backend = "opencv"
device_index = 0
wait_key_ms = 1

[paths]
c2p_map = ""
p2c_map = "maps/p2c.npy"
warp_method = "p2c"
target_image_dir = "data/target_images"
linear_pattern_dir = "data/linear_proj_patterns"
inv_gamma_pattern_dir = "data/inv_gamma_proj_patterns"
captured_image_dir = "data/captured_images"
compensation_image_dir = "data/compensation_images"
inv_gamma_comp_dir = "data/inv_gamma_comp_images"

[compensation]
num_divisions = 2
use_gpu = false
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    for func_name in (
        "namedWindow",
        "setWindowProperty",
        "moveWindow",
        "imshow",
        "destroyWindow",
    ):
        monkeypatch.setattr(sample.cv2, func_name, lambda *args, **kwargs: None)
    monkeypatch.setattr(sample.cv2, "waitKey", lambda *args, **kwargs: 0)

    captured_images = iter(
        sample.generate_projection_patterns(
            width,
            height,
            num_divisions=2,
            dtype=np.dtype(np.uint8),
        )
    )
    monkeypatch.setattr(
        sample,
        "capture_image",
        lambda: next(captured_images, None),
    )

    sample.main(["sample.py", "--config", str(config_path)])

    assert (data_dir / "color_mixing_matrices.npy").exists()
    assert len(list((data_dir / "linear_proj_patterns").glob("pattern_*.png"))) == 8
    assert (
        len(list((data_dir / "inv_gamma_proj_patterns").glob("inv_gamma_pattern_*.png")))
        == 8
    )
    assert len(list((data_dir / "captured_images").glob("captured_image_*.png"))) == 8
    assert (data_dir / "compensation_images" / "compensation_image_00.png").exists()
    assert (data_dir / "inv_gamma_comp_images" / "inv_gamma_comp_image_00.png").exists()
