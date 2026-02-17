import os
import uuid
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ocr_pipeline import OCRPipeline

app = Flask(__name__)
CORS(app)  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, "result")
CONFIG_PATH = os.path.join(BASE_DIR, "config", "ocr_config.json")
PARAMS_PATH = os.path.join(BASE_DIR, "config", "all_params.json")
GOOGLE_CREDS = os.path.join(BASE_DIR, "credentials", "google_vision_key.json")
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
            if key and (key not in os.environ or not os.environ.get(key)):
                os.environ[key] = value

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
load_env_file(DEFAULT_ENV_FILE)
DEFAULT_OCR_SERVICE = os.getenv("MV0_DEFAULT_OCR_SERVICE", "ai_models")

print("Initializing OCR Pipeline...")
try:
    pipeline = OCRPipeline(
        config_path=CONFIG_PATH,
        params_path=PARAMS_PATH,
        ocr_service=DEFAULT_OCR_SERVICE,
        google_credentials_path=GOOGLE_CREDS,
        local_lens_url=os.getenv("MV0_LOCAL_LENS_URL"),
    )
    print("Pipeline initialized successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not initialize pipeline. {e}")
    pipeline = None

@app.route('/process-lab', methods=['POST'])
def process_lab_report():
    if not pipeline:
        return jsonify({"error": "Server is not ready (Pipeline failed to load)"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        file_path = None
        try:
            requested_service = request.form.get("ocr_service", DEFAULT_OCR_SERVICE)
            original_filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{original_filename}"
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            
            file.save(file_path)
            print(f"File saved to: {file_path}")
            print(f"Processing with ocr_service: {requested_service}")
            request_pipeline = OCRPipeline(
                config_path=CONFIG_PATH,
                params_path=PARAMS_PATH,
                ocr_service=requested_service,
                google_credentials_path=GOOGLE_CREDS,
                local_lens_url=os.getenv("MV0_LOCAL_LENS_URL"),
            )
            result = request_pipeline.run(file_path)

            output_name = datetime.now().strftime("tr_%Y-%m-%d&%H-%M-%S.json")
            output_path = os.path.join(RESULT_FOLDER, output_name)
            with open(output_path, "w", encoding="utf-8") as output_file:
                json.dump(result, output_file, ensure_ascii=False, indent=2)
            print(f"Result saved to: {output_path}")
            
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File deleted: {file_path}")
            
            return jsonify({
                "status": "success",
                "data": result,
                "saved_result_path": output_path,
                "ocr_service_requested": requested_service,
                "ocr_service_used": result.get("ocr_service_used"),
            })

        except Exception as e:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            print(f"Error processing file: {e}")
            return jsonify({"error": str(e), "status": "failed"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
