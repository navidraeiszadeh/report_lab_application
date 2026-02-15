import argparse
import json
import os
import sys

from ocr_pipeline import OCRPipeline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "config", "ocr_config.json")
DEFAULT_PARAMS_PATH = os.path.join(BASE_DIR, "config", "all_params.json")
DEFAULT_GOOGLE_CREDS = os.path.join(BASE_DIR, "credentials", "google_vision_key.json")
DEFAULT_ENV_FILE = os.path.join(BASE_DIR, ".env")


def load_env_file(env_path: str) -> None:
    if not env_path or not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Allow .env to populate empty exported vars from shell/session.
            if key and (key not in os.environ or not os.environ.get(key)):
                os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local lab OCR pipeline")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument(
        "--ocr-service",
        default="google_lens",
        choices=["google_lens", "alefba", "ai_models"],
        help="OCR service",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to ocr_config.json")
    parser.add_argument("--params", default=DEFAULT_PARAMS_PATH, help="Path to all_params.json")
    parser.add_argument(
        "--google-creds",
        default=DEFAULT_GOOGLE_CREDS,
        help="Path to Google Vision service account JSON",
    )
    parser.add_argument("--alefba-url", default=None, help="Alefba OCR URL")
    parser.add_argument("--alefba-token", default=None, help="Alefba OCR token")
    parser.add_argument("--ai-ocr-key", default=None, help="AI OCR API key (AvalAI)")
    parser.add_argument("--ai-ocr-base-url", default=None, help="AI OCR base URL (AvalAI)")
    parser.add_argument("--ai-ocr-model", default=None, help="AI OCR model name")
    parser.add_argument("--openai-key", default=None, help="OpenAI key")
    parser.add_argument("--openai-base-url", default=None, help="OpenAI base URL")
    parser.add_argument("--openai-model", default=None, help="OpenAI model name")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE, help="Path to .env file")
    parser.add_argument("--output", default=None, help="Output JSON file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)

    pipeline = OCRPipeline(
        config_path=args.config,
        params_path=args.params,
        ocr_service=args.ocr_service,
        google_credentials_path=args.google_creds,
        alefba_url=args.alefba_url,
        alefba_token=args.alefba_token,
        ai_ocr_key=args.ai_ocr_key,
        ai_ocr_base_url=args.ai_ocr_base_url,
        ai_ocr_model=args.ai_ocr_model,
        openai_key=args.openai_key,
        openai_base_url=args.openai_base_url,
        openai_model=args.openai_model,
    )

    result = pipeline.run(args.pdf)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved output to {args.output}")
    else:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
