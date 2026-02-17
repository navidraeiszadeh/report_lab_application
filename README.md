
Project layout
- `run_ocr.py`
- `ocr_pipeline.py`
- `config/ocr_config.json`
- `config/all_params.json`
- `credentials/google_vision_key.json` 
- `.env` 

Setup
- `cd lab_ocr_local`
- `python -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

Set API key and model (local only)
1. Create env file from template

2. Edit `.env` and set:
- `MV0_AI_KEY=...`
- `MV0_AI_BASE_URL=...`
- `MV0_AI_MODEL_NAME=...`
- `MV0_LOCAL_LENS_URL=http://vision-api.mamania.me:8000/predict/pdf/`
- `MV0_DEFAULT_OCR_SERVICE=ai_models`

Google Vision credentials
- Put your service account JSON at:
  - `credentials/google_vision_key.json`
- This file is gitignored by default.

Run
- `python run_ocr.py --pdf lab.pdf --output /result/lab_output2.json`

OCR dynamic fallback
- If `--ocr-service google_lens` returns empty text or fails, pipeline automatically retries using `ai_models`.
- Logs show request/response flow and selected/fallback services.

Optional: AI models OCR via AvalAI
- Set in `.env`:
  - `MV0_OCR_AI_KEY=...`
  - `MV0_OCR_AI_BASE_URL=https://api.avalai.ir/v1`
  - `MV0_OCR_AI_MODEL=...`
- Run:
  - `python run_ocr.py --pdf lab.pdf --ocr-service ai_models --output /tmp/lab_output.json`

Optional: Local lens OCR service
- Set in `.env`:
  - `MV0_LOCAL_LENS_URL=http://vision-api.mamania.me:8000/predict/pdf/`
- Run:
  - `python run_ocr.py --pdf lab.pdf --ocr-service local_lens_service --output ./lab_output.json`
- Service example:
```bash
curl --location 'http://vision-api.mamania.me:8000/predict/pdf/' \
```

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

api.py is a Flask api for these services
- run "`python api.py`"
- set the url in a front project based of ip
- find the ip with "`ipconfig`"
- upload endpoint: `POST /process-lab` (form fields: `file`, optional `ocr_service`)
- accepted `ocr_service`: `google_lens`, `ai_models`, `local_lens_service`, `alefba`
