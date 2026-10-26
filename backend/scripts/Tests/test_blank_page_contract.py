import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from backend.scripts import ocrFullBanAn
from backend.scripts.Evaluation.benchmarking.independent_benchmark import (
    canonical_json_fingerprint,
    inventory_judgment_sources,
)
from backend.scripts.Evaluation.benchmarking.source_intake import (
    SourceIntakeError,
    validate_structured_judgment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = (
    PROJECT_ROOT
    / "backend/dataLaw/independent_intake/structured_judgment.schema.json"
)


def _payload(pages: list[dict]) -> dict:
    usable_text = "\n".join(page["text"] for page in pages if page["text"])
    return {
        "judgment_name": "decision.pdf",
        "page_count": len(pages),
        "pages": pages,
        "sections": {
            "header": usable_text,
            "facts": "",
            "reasoning": "",
            "decision": "",
        },
    }


def _schema_validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.mark.parametrize(
    ("status", "text", "valid"),
    [
        (None, "Legacy usable text", True),
        (None, "", False),
        ("native_text", "Native usable text", True),
        ("native_text", "", False),
        ("ocr_text", "OCR usable text", True),
        ("ocr_text", "", False),
        ("empty_after_native_and_ocr", "", True),
        ("empty_after_native_and_ocr", "Unexpected text", False),
        ("unknown", "Text", False),
    ],
    ids=[
        "legacy-nonempty",
        "legacy-empty",
        "native-nonempty",
        "native-empty",
        "ocr-nonempty",
        "ocr-empty",
        "explicit-blank-empty",
        "explicit-blank-nonempty",
        "unknown-status",
    ],
)
def test_schema_and_runtime_enforce_page_status_text_contract(status, text, valid):
    page_number = 2 if not text else 1
    page = {"page": page_number, "text": text}
    if status is not None:
        page["extraction_status"] = status
    pages = [page]
    if not text:
        pages.insert(
            0,
            {
                "page": 1,
                "text": "Usable companion page",
                "extraction_status": "native_text",
            },
        )
    payload = _payload(pages)
    schema_errors = list(_schema_validator().iter_errors(payload))

    if valid:
        assert schema_errors == []
        validate_structured_judgment(payload, "decision.pdf")
    else:
        assert schema_errors
        with pytest.raises(SourceIntakeError):
            validate_structured_judgment(payload, "decision.pdf")


def test_document_with_usable_and_explicit_blank_pages_is_valid():
    payload = _payload(
        [
            {
                "page": 1,
                "text": "Nội dung bản án hợp lệ",
                "extraction_status": "native_text",
            },
            {
                "page": 2,
                "text": "",
                "extraction_status": "empty_after_native_and_ocr",
            },
        ]
    )

    assert payload["page_count"] == len(payload["pages"]) == 2
    assert [page["page"] for page in payload["pages"]] == [1, 2]
    assert list(_schema_validator().iter_errors(payload)) == []
    validate_structured_judgment(payload, "decision.pdf")


def test_document_with_only_explicit_blank_pages_is_invalid():
    payload = _payload(
        [
            {
                "page": 1,
                "text": "",
                "extraction_status": "empty_after_native_and_ocr",
            },
            {
                "page": 2,
                "text": "",
                "extraction_status": "empty_after_native_and_ocr",
            },
        ]
    )

    assert list(_schema_validator().iter_errors(payload))
    with pytest.raises(SourceIntakeError, match="usable document text"):
        validate_structured_judgment(payload, "decision.pdf")


@pytest.mark.parametrize(
    "mutation",
    ["page_count", "duplicate_page", "missing_page"],
)
def test_runtime_rejects_page_count_and_sequence_violations(mutation):
    payload = _payload(
        [
            {"page": 1, "text": "Trang một", "extraction_status": "native_text"},
            {"page": 2, "text": "Trang hai", "extraction_status": "ocr_text"},
        ]
    )
    if mutation == "page_count":
        payload["page_count"] = 3
    elif mutation == "duplicate_page":
        payload["pages"][1]["page"] = 1
    else:
        payload["pages"][1]["page"] = 3

    with pytest.raises(SourceIntakeError, match="page_count|consecutive"):
        validate_structured_judgment(payload, "decision.pdf")


class _FakePlumberPage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class _FakePlumberDocument:
    def __init__(self, text):
        self.pages = [_FakePlumberPage(text)]

    def close(self):
        return None


class _FakeBitmap:
    def to_pil(self):
        return object()


class _FakePdfiumPage:
    def render(self, scale):
        assert scale == 2
        return _FakeBitmap()


class _FakePdfiumDocument:
    def __init__(self, _path):
        pass

    def __getitem__(self, index):
        assert index == 0
        return _FakePdfiumPage()


def _install_extraction_fakes(monkeypatch, native_text, ocr_result=None, ocr_error=None):
    import pdfplumber
    import pypdfium2
    import pytesseract

    ocr_calls = []
    monkeypatch.setattr(
        pdfplumber,
        "open",
        lambda _path: _FakePlumberDocument(native_text),
    )
    monkeypatch.setattr(pypdfium2, "PdfDocument", _FakePdfiumDocument)

    def fake_ocr(image, lang):
        ocr_calls.append((image, lang))
        if ocr_error is not None:
            raise ocr_error
        return ocr_result

    monkeypatch.setattr(pytesseract, "image_to_string", fake_ocr)
    return ocr_calls


def test_extractor_marks_native_page_without_calling_ocr(tmp_path, monkeypatch):
    calls = _install_extraction_fakes(monkeypatch, "N" * 120)

    pages = ocrFullBanAn.extract_full_pages(tmp_path / "decision.pdf")

    assert pages == [
        {"page": 1, "text": "N" * 120, "extraction_status": "native_text"}
    ]
    assert calls == []


def test_extractor_marks_ocr_recovered_page(tmp_path, monkeypatch):
    calls = _install_extraction_fakes(
        monkeypatch,
        "",
        ocr_result="Nội dung OCR hợp lệ",
    )

    pages = ocrFullBanAn.extract_full_pages(tmp_path / "decision.pdf")

    assert pages == [
        {
            "page": 1,
            "text": "Nội dung OCR hợp lệ",
            "extraction_status": "ocr_text",
        }
    ]
    assert len(calls) == 1
    assert calls[0][1] == "vie"


def test_extractor_marks_true_blank_only_after_successful_ocr(tmp_path, monkeypatch):
    calls = _install_extraction_fakes(monkeypatch, "", ocr_result=" \n\t")

    pages = ocrFullBanAn.extract_full_pages(tmp_path / "decision.pdf")

    assert pages == [
        {
            "page": 1,
            "text": "",
            "extraction_status": "empty_after_native_and_ocr",
        }
    ]
    assert len(calls) == 1


def test_ocr_runtime_error_fails_extraction_instead_of_classifying_blank(
    tmp_path, monkeypatch
):
    calls = _install_extraction_fakes(
        monkeypatch,
        "",
        ocr_error=RuntimeError("diagnostic OCR failure"),
    )

    pages = ocrFullBanAn.extract_full_pages(tmp_path / "decision.pdf")

    assert pages == []
    assert len(calls) == 1


def test_all_historical_structured_judgments_remain_legacy_compatible():
    paths = sorted(
        (PROJECT_ROOT / "backend/data/structured_judgments").glob("*.json")
    )
    assert len(paths) == 107
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert all("extraction_status" not in page for page in payload["pages"])

    inventory = inventory_judgment_sources(PROJECT_ROOT)
    assert len(inventory) == 107
    assert canonical_json_fingerprint(inventory) == (
        "fd6a6f05e4b71a2c499584fc0296cf25cedf5eac958aea6d079bf449c864f754"
    )
