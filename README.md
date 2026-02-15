Local Lab OCR Pipeline (Standalone)

This folder is now self-contained and can be published as a separate project.

Project layout
- `run_ocr.py`
- `ocr_pipeline.py`
- `config/ocr_config.json`
- `config/all_params.json`
- `credentials/google_vision_key.json` (you create this locally)
- `.env` (you create this locally)

Setup
- `cd lab_ocr_local`
- `python -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

Set API key and model (local only)
1. Create env file from template:
- `cp .env.example .env`

2. Edit `.env` and set:
- `MV0_AI_KEY=...`
- `MV0_AI_BASE_URL=...`
- `MV0_AI_MODEL_NAME=...`

Google Vision credentials
- Put your service account JSON at:
  - `credentials/google_vision_key.json`
- This file is gitignored by default.

Run
- `python run_ocr.py --pdf lab.pdf --output /tmp/lab_output.json`

Optional: AI models OCR via AvalAI
- Set in `.env`:
  - `MV0_OCR_AI_KEY=...`
  - `MV0_OCR_AI_BASE_URL=https://api.avalai.ir/v1`
  - `MV0_OCR_AI_MODEL=...`
- Run:
  - `python run_ocr.py --pdf lab.pdf --ocr-service ai_models --output /tmp/lab_output.json`

Optional: Alefba backend
- Set in `.env`:
  - `MV0_ALEFBA_URL=...`
  - `MV0_ALEFBA_TOKEN=...`
- Run:
  - `python run_ocr.py --pdf lab.pdf --ocr-service alefba --output /tmp/lab_output.json`

Jupyter
```python
from ocr_pipeline import OCRPipeline

pipeline = OCRPipeline(
    config_path="config/ocr_config.json",
    params_path="config/all_params.json",
    ocr_service="google_lens",
    google_credentials_path="credentials/google_vision_key.json",
)

result = pipeline.run("lab.pdf")
```

Publishing as separate project
- Keep `.env` and `credentials/*.json` out of git (already in `.gitignore`).
- Commit code + `config/*.json` + `.env.example`.
- Add your own project `LICENSE` and `pyproject.toml` if you want packaging/distribution.
