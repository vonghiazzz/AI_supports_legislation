import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from backend.scripts.Evaluation.benchmarking import source_intake


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_library_import_ignores_unrelated_app_env_and_has_no_side_effects(tmp_path):
    code = textwrap.dedent(
        """
        import builtins
        import json
        import pathlib
        import socket
        import sys
        import urllib.request

        workdir = pathlib.Path.cwd()
        before = sorted(path.relative_to(workdir).as_posix() for path in workdir.rglob("*"))

        def forbidden_network(*args, **kwargs):
            raise AssertionError("network access attempted during extraction-library import")

        socket.create_connection = forbidden_network
        socket.socket.connect = forbidden_network
        urllib.request.urlopen = forbidden_network

        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "app.config" or name.startswith("google.generativeai"):
                raise AssertionError(f"forbidden import-time dependency: {name}")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = guarded_import
        from backend.scripts import ocrFullBanAn

        after = sorted(path.relative_to(workdir).as_posix() for path in workdir.rglob("*"))
        assert before == after
        assert "app.config" not in sys.modules
        assert "google.generativeai" not in sys.modules
        assert callable(ocrFullBanAn.process_judgment_file)
        print(json.dumps({"imported": True, "files": after}))
        """
    )
    environment = os.environ.copy()
    environment.update(
        {
            "DEBUG": "release",
            "DATABASE_PORT": "not-an-integer",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(PROJECT_ROOT),
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"imported": True, "files": []}


def test_process_judgment_file_remains_callable_and_non_overwriting(
    tmp_path, monkeypatch
):
    from backend.scripts import ocrFullBanAn

    calls = []

    def fake_extract(pdf_path, force_ocr=False, tesseract_path=None):
        calls.append((pdf_path, force_ocr, tesseract_path))
        return [
            {
                "page": 1,
                "text": "Nội dung bản án hợp lệ",
                "extraction_status": "native_text",
            }
        ]

    monkeypatch.setattr(ocrFullBanAn, "extract_full_pages", fake_extract)
    input_path = tmp_path / "decision.pdf"
    output_dir = tmp_path / "structured"

    assert ocrFullBanAn.process_judgment_file(input_path, output_dir) is True
    assert ocrFullBanAn.process_judgment_file(input_path, output_dir) is False
    assert calls == [
        (input_path, False, None),
        (input_path, False, None),
    ]
    payload = json.loads(
        (output_dir / "decision_structured.json").read_text(encoding="utf-8")
    )
    assert payload["judgment_name"] == "decision.pdf"
    assert payload["pages"] == [
        {
            "page": 1,
            "text": "Nội dung bản án hợp lệ",
            "extraction_status": "native_text",
        }
    ]


def test_process_judgment_file_forwards_explicit_tesseract_path(
    tmp_path, monkeypatch
):
    from backend.scripts import ocrFullBanAn

    observed = []

    def fake_extract(pdf_path, force_ocr=False, tesseract_path=None):
        observed.append((force_ocr, tesseract_path))
        return [
            {
                "page": 1,
                "text": "Nội dung bản án hợp lệ",
                "extraction_status": "native_text",
            }
        ]

    monkeypatch.setattr(ocrFullBanAn, "extract_full_pages", fake_extract)
    assert ocrFullBanAn.process_judgment_file(
        tmp_path / "decision.pdf",
        tmp_path / "structured",
        force_ocr=True,
        tesseract_path="/local/bin/tesseract",
    )
    assert observed == [(True, "/local/bin/tesseract")]


def test_source_intake_resolves_real_safe_library_entrypoint(tmp_path, monkeypatch):
    monkeypatch.setenv("DEBUG", "release")
    from backend.scripts import ocrFullBanAn

    calls = []

    def fake_process(
        input_path,
        output_dir,
        *,
        force_ocr,
        overwrite,
        output_dir_fd,
        original_filename,
    ):
        calls.append(
            (
                input_path,
                output_dir,
                force_ocr,
                overwrite,
                output_dir_fd,
                original_filename,
            )
        )
        return True

    monkeypatch.setattr(ocrFullBanAn, "process_judgment_file", fake_process)
    raw_path = tmp_path / "decision.pdf"
    output_dir = tmp_path / "structured"
    raw_path.write_bytes(b"not-read-by-mocked-adapter")
    output_dir.mkdir()
    raw_fd = os.open(raw_path, os.O_RDONLY)
    output_fd = os.open(output_dir, os.O_RDONLY)
    try:
        assert source_intake._run_local_extractor(
            raw_fd, output_fd, "decision.pdf", False
        )
        assert calls == [
            (
                Path(f"/dev/fd/{raw_fd}"),
                Path("."),
                False,
                False,
                output_fd,
                "decision.pdf",
            )
        ]
    finally:
        os.close(output_fd)
        os.close(raw_fd)


def test_cli_passes_specific_tesseract_path_without_app_settings(
    tmp_path, monkeypatch
):
    from backend.scripts import ocrFullBanAn

    input_path = tmp_path / "decision.pdf"
    input_path.write_bytes(b"not-read-by-mocked-cli")
    observed = []

    def fake_process(pdf_path, output_dir, force_ocr, **kwargs):
        observed.append((pdf_path, output_dir, force_ocr, kwargs))
        return True

    monkeypatch.setattr(
        ocrFullBanAn, "_load_cli_tesseract_path", lambda: "/cli/bin/tesseract"
    )
    monkeypatch.setattr(ocrFullBanAn, "process_judgment_file", fake_process)
    monkeypatch.setattr(
        sys,
        "argv",
        ["ocrFullBanAn.py", str(input_path), "-o", str(tmp_path / "output")],
    )

    ocrFullBanAn.main()

    assert observed == [
        (
            input_path,
            tmp_path / "output",
            False,
            {"overwrite": False, "tesseract_path": "/cli/bin/tesseract"},
        )
    ]
