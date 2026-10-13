# Legacy Scripts

These scripts are preserved for historical provenance and reproducibility.

They are not the active law-processing pipeline.

Active generalized data pipeline:
- `../Flow/01_extract_laws.py`
- `../Flow/02_transform_laws.py`
- `../Flow/03_load_vectors.py`
- `../Flow/run_pipeline.py`

Important:
- Do not use legacy scripts to regenerate the frozen research corpus.
- Do not treat `Flow/03_load_vectors.py` as the canonical baseline runtime artifact builder until runtime/baseline representation and artifact invariants are explicitly aligned.
