import copy
from pathlib import Path

import pytest
import yaml

from backend.scripts.Evaluation.experiments import baseline as baseline_experiment


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "backend" / "scripts" / "Evaluation" / "configs"
DATA_DIR = PROJECT_ROOT / "backend" / "data"
LAWS_CONFIG_PATH = PROJECT_ROOT / "backend" / "dataLaw" / "laws_config.json"
BASELINE_CONFIG_PATH = CONFIG_DIR / "baseline_d0.yaml"
EXP02_D0_CONFIG_PATH = CONFIG_DIR / "exp02_d0.yaml"
EXP02_D1_CONFIG_PATH = CONFIG_DIR / "exp02_d1.yaml"


def _load_config(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _set_nested(config, path, value):
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def test_frozen_baseline_accepts_d0():
    protocol = baseline_experiment._validate_baseline_protocol(
        _load_config(BASELINE_CONFIG_PATH)
    )

    assert protocol["protocol_id"] == "baseline_d0_v2"
    assert protocol["document_representation"] == "d0"
    assert protocol["experiment_arm"] is None
    assert baseline_experiment._run_identity(protocol) == "baseline_d0_v2"
    assert baseline_experiment._experiment_metadata(protocol) == {}


def test_frozen_baseline_rejects_d1():
    config = _load_config(BASELINE_CONFIG_PATH)
    config["evaluation"]["document_representation"] = "d1"

    with pytest.raises(
        ValueError, match="baseline_d0_v2 requires document_representation=d0"
    ):
        baseline_experiment._validate_baseline_protocol(config)


@pytest.mark.parametrize(
    ("config_path", "expected_arm"),
    [
        (EXP02_D0_CONFIG_PATH, "d0"),
        (EXP02_D1_CONFIG_PATH, "d1"),
    ],
)
def test_exp02_arms_are_valid_and_explicit(config_path, expected_arm):
    protocol = baseline_experiment._validate_baseline_protocol(
        _load_config(config_path)
    )

    assert protocol["protocol_id"] == "exp02_d0_vs_d1_v1"
    assert protocol["experiment_id"] == "exp02_d0_vs_d1_v1"
    assert protocol["experiment_arm"] == expected_arm
    assert protocol["control_protocol"] == "baseline_d0_v2"
    assert protocol["document_representation"] == expected_arm
    assert baseline_experiment._run_identity(protocol) == (
        f"exp02_d0_vs_d1_v1_{expected_arm}"
    )
    assert baseline_experiment._experiment_metadata(protocol) == {
        "experiment_id": "exp02_d0_vs_d1_v1",
        "experiment_arm": expected_arm,
        "control_protocol": "baseline_d0_v2",
    }


def test_exp02_configs_differ_only_by_arm_and_document_representation():
    d0_config = _load_config(EXP02_D0_CONFIG_PATH)
    d1_config = _load_config(EXP02_D1_CONFIG_PATH)

    d0_config["protocol"]["experiment_arm"] = "<arm>"
    d1_config["protocol"]["experiment_arm"] = "<arm>"
    d0_config["evaluation"]["document_representation"] = "<representation>"
    d1_config["evaluation"]["document_representation"] = "<representation>"

    assert d0_config == d1_config


@pytest.mark.parametrize("config_path", [EXP02_D0_CONFIG_PATH, EXP02_D1_CONFIG_PATH])
def test_exp02_arms_freeze_all_controlled_settings(config_path):
    config = _load_config(config_path)
    protocol = baseline_experiment._validate_baseline_protocol(config)

    assert protocol["query_representation"] == "q0"
    assert protocol["temporal_policy"] == "none"
    assert protocol["bm25_tokenizer"] == "whitespace"
    assert protocol["fusion_method"] == "minmax"
    assert protocol["static_w_bm25"] == 0.5
    assert protocol["tie_break_policy"] == "standard_law_id_asc"
    assert config["evaluation"]["pool_k"] == 30
    assert config["evaluation"]["k"] == 20
    assert config["models"] == {
        "embedding_model": "keepitreal/vietnamese-sbert",
        "embedding_revision": "a9467ef2ef47caa6448edeabfd8e5e5ce0fa2a23",
        "local_files_only": True,
        "rerank_model": "itdainb/PhoRanker",
    }
    assert config["corpus"]["expected_total"] == 1811
    assert config["corpus"]["expected_counts"] == {
        "bo_luat_dan_su_2015": 689,
        "bo_luat_to_tung_dan_su_2015": 517,
        "luat_dat_dai_2013": 212,
        "luat_dat_dai_2024": 260,
        "luat_hon_nhan_gia_dinh_2014": 133,
    }


@pytest.mark.parametrize(
    ("path", "invalid_value", "message"),
    [
        (("evaluation", "query_representation"), "q1", "query_representation=q0"),
        (("evaluation", "temporal_filter"), "judgment_year", "temporal_filter=none"),
        (("evaluation", "pool_k"), 31, "pool_k=30"),
        (("evaluation", "k"), 21, "retrieval depth k=20"),
        (("bm25", "tokenizer"), "legacy_optional_pyvi", "bm25.tokenizer=whitespace"),
        (("fusion", "method"), "rrf", "minmax fusion"),
        (("fusion", "static_w_bm25"), 0.6, "static_w_bm25=0.5"),
        (("fusion", "tie_break_policy"), "none", "fusion tie-break"),
        (("models", "embedding_model"), "other/model", "embedding_model"),
        (("models", "embedding_revision"), "other-revision", "embedding revision"),
        (("corpus", "expected_total"), 1812, "corpus total=1811"),
    ],
)
def test_exp02_validator_rejects_changed_controls(path, invalid_value, message):
    config = copy.deepcopy(_load_config(EXP02_D0_CONFIG_PATH))
    _set_nested(config, path, invalid_value)

    with pytest.raises(ValueError, match=message):
        baseline_experiment._validate_baseline_protocol(config)


def test_exp02_validator_rejects_arm_representation_mismatch():
    config = _load_config(EXP02_D0_CONFIG_PATH)
    config["evaluation"]["document_representation"] = "d1"

    with pytest.raises(ValueError, match="must match protocol.experiment_arm"):
        baseline_experiment._validate_baseline_protocol(config)


def test_baseline_source_provenance_excludes_exp02_test():
    source_paths = baseline_experiment._source_paths(
        BASELINE_CONFIG_PATH, PROJECT_ROOT, "baseline_d0_v2"
    )

    assert "backend/scripts/Evaluation/configs/baseline_d0.yaml" in source_paths
    assert "backend/scripts/Tests/test_baseline_protocol.py" in source_paths
    assert "backend/scripts/Tests/test_exp02_protocol.py" not in source_paths


@pytest.mark.parametrize("config_path", [EXP02_D0_CONFIG_PATH, EXP02_D1_CONFIG_PATH])
def test_source_provenance_tracks_the_actual_exp02_config(config_path):
    source_paths = baseline_experiment._source_paths(
        config_path, PROJECT_ROOT, "exp02_d0_vs_d1_v1"
    )
    expected = str(config_path.relative_to(PROJECT_ROOT))

    assert expected in source_paths
    assert "backend/scripts/Evaluation/configs/baseline_d0.yaml" not in source_paths
    assert "backend/scripts/Tests/test_baseline_protocol.py" in source_paths
    assert "backend/scripts/Tests/test_exp02_protocol.py" in source_paths

    config_provenance = baseline_experiment._config_provenance(config_path)
    assert config_provenance["path"] == str(config_path.resolve())
    assert len(config_provenance["sha256"]) == 64


def test_exp02_uses_the_frozen_corpus_fingerprint():
    config = _load_config(EXP02_D0_CONFIG_PATH)
    _, inventory, total = baseline_experiment.load_frozen_baseline_corpus(
        LAWS_CONFIG_PATH, DATA_DIR, config["corpus"]
    )

    assert total == 1811
    assert baseline_experiment._combined_corpus_fingerprint(inventory) == (
        "66fb0b20e0505993ed616f864ece99ecfad801de98b4ba3a37d20a3a7bf55e58"
    )
