from pathlib import Path
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external.GrayCode.src.python import pipeline as graycode_pipeline


def _app_config(
    *,
    c2p_input: str = "result_c2p.npy",
    p2c_input: str = "result_p2c.npy",
    interpolation_method: str = "delaunay",
):
    return SimpleNamespace(
        pipeline=SimpleNamespace(
            default_input_file=c2p_input,
            default_p2c_input_file=p2c_input,
            default_interpolation_method=interpolation_method,
        )
    )


def test_run_graycode_pipeline_runs_p2c_interpolation(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.chdir(tmp_path)
    (tmp_path / "custom_p2c.npy").write_bytes(b"placeholder")
    monkeypatch.setattr(
        graycode_pipeline,
        "get_config",
        lambda: _app_config(
            c2p_input="custom_c2p.npy",
            p2c_input="custom_p2c.npy",
            interpolation_method="delaunay",
        ),
    )
    monkeypatch.setattr(
        graycode_pipeline.gen_graycode,
        "main",
        lambda argv: calls.append(("gen", argv)),
    )
    monkeypatch.setattr(
        graycode_pipeline.cap_graycode,
        "main",
        lambda argv: calls.append(("cap", argv)),
    )

    def fake_decode(argv):
        calls.append(("decode", argv))
        return (720, 1280)

    monkeypatch.setattr(graycode_pipeline.decode, "main", fake_decode)
    monkeypatch.setattr(
        graycode_pipeline.interpolate_c2p,
        "main",
        lambda argv: calls.append(("c2p", argv)),
    )
    monkeypatch.setattr(
        graycode_pipeline.interpolate_p2c,
        "main",
        lambda argv: calls.append(("p2c", argv)),
    )

    cfg = graycode_pipeline.GraycodePipelineConfig(
        proj_height=1080,
        proj_width=1920,
        height_step=2,
        width_step=3,
        window_pos_x=10,
        window_pos_y=20,
    )

    graycode_pipeline.run_graycode_pipeline(cfg)

    assert calls == [
        ("gen", ["gen_graycode.py", "1080", "1920", "2", "3"]),
        ("cap", ["cap_graycode.py", "10", "20"]),
        ("decode", ["decode.py", "1080", "1920", "2", "3"]),
        (
            "c2p",
            ["interpolate_c2p.py", "custom_c2p.npy", "720", "1280", "delaunay"],
        ),
        ("p2c", ["interpolate_p2c.py", "custom_p2c.npy", "1080", "1920"]),
    ]


def test_run_graycode_pipeline_skips_p2c_when_input_file_is_missing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        graycode_pipeline,
        "get_config",
        lambda: _app_config(p2c_input="missing_p2c.npy"),
    )
    monkeypatch.setattr(graycode_pipeline.gen_graycode, "main", lambda argv: None)
    monkeypatch.setattr(graycode_pipeline.cap_graycode, "main", lambda argv: None)
    monkeypatch.setattr(graycode_pipeline.decode, "main", lambda argv: (100, 200))
    monkeypatch.setattr(
        graycode_pipeline.interpolate_c2p,
        "main",
        lambda argv: calls.append(("c2p", argv)),
    )
    monkeypatch.setattr(
        graycode_pipeline.interpolate_p2c,
        "main",
        lambda argv: calls.append(("p2c", argv)),
    )

    cfg = graycode_pipeline.GraycodePipelineConfig(
        proj_height=480,
        proj_width=640,
    )

    graycode_pipeline.run_graycode_pipeline(cfg)

    captured = capsys.readouterr()

    assert calls == [
        ("c2p", ["interpolate_c2p.py", "result_c2p.npy", "100", "200", "delaunay"])
    ]
    assert "Skipped p2c interpolate" in captured.out
