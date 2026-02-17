import json
import os
import time
import base64
import re
import sys
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class OCRConfig:
    system_message: str
    human_message: str
    patient_info_schema: Dict[str, Any]
    test_result_schema: Dict[str, Any]
    name_mapping: Dict[str, List[str]]
    result_mapping: Dict[str, Dict[str, List[str]]]


@dataclass
class ParamType:
    en_name: str
    abbreviation: Optional[str]
    representation_type: str
    unit: Optional[str]
    lb: Optional[float]
    ub: Optional[float]
    fix_point: int
    ordered_values: Optional[List[str]]


class OCRPipeline:
    def __init__(
        self,
        config_path: str,
        params_path: str,
        ocr_service: str = "google_lens",
        google_credentials_path: Optional[str] = None,
        alefba_url: Optional[str] = None,
        alefba_token: Optional[str] = None,
        local_lens_url: Optional[str] = None,
        ai_ocr_key: Optional[str] = None,
        ai_ocr_base_url: Optional[str] = None,
        ai_ocr_model: Optional[str] = None,
        openai_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        openai_model: Optional[str] = None,
    ):
        self.config_path = config_path
        self.params_path = params_path
        self.ocr_service = ocr_service
        self.google_credentials_path = google_credentials_path
        self.alefba_url = alefba_url
        self.alefba_token = alefba_token
        self.local_lens_url = local_lens_url
        self.ai_ocr_key = ai_ocr_key
        self.ai_ocr_base_url = ai_ocr_base_url
        self.ai_ocr_model = ai_ocr_model
        self.openai_key = openai_key
        self.openai_base_url = openai_base_url
        self.openai_model = openai_model
        self.last_ocr_service_used = ocr_service

        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        self.logger = logging.getLogger(__name__)

        self.config = self._load_config(config_path)
        self.param_lookup = self._load_params(params_path)

    def run(self, pdf_path: str) -> Dict[str, Any]:
        pages_text = self._extract_text(pdf_path)
        full_text = "\n\n".join(pages_text)

        extracted = self._text_to_json(full_text)
        mapped_data, unmapped_data = self._map_json(extracted)
        ok_parameters, not_ok_params = self._process_mapped(mapped_data, unmapped_data)

        return {
            "ocr_service_requested": self.ocr_service,
            "ocr_service_used": self.last_ocr_service_used,
            "ocr_text_pages": pages_text,
            "extracted_data": extracted,
            "mapped_data": mapped_data,
            "unmapped_data": unmapped_data,
            "ok_parameters": ok_parameters,
            "not_ok_params": not_ok_params,
        }

    def _load_config(self, config_path: str) -> OCRConfig:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)

        parsed: Dict[str, Any] = {}
        for item in raw_items:
            key = item.get("key")
            value = item.get("value")
            value_type = item.get("value_type")
            if value_type == 0:
                parsed[key] = json.loads(value)
            else:
                parsed[key] = value

        return OCRConfig(
            system_message=parsed.get("system message", ""),
            human_message=parsed.get("human message", ""),
            patient_info_schema=parsed.get("patient info schema", {}),
            test_result_schema=parsed.get("test result schema", {}),
            name_mapping=parsed.get("name mapping", {}),
            result_mapping=parsed.get("result mapping", {}),
        )

    def _load_params(self, params_path: str) -> Dict[str, ParamType]:
        with open(params_path, "r", encoding="utf-8") as f:
            raw_params = json.load(f)

        lookup: Dict[str, ParamType] = {}
        for item in raw_params:
            param = ParamType(
                en_name=item.get("en_name"),
                abbreviation=item.get("abbreviation"),
                representation_type=str(item.get("representation_type", "")).lower(),
                unit=item.get("unit"),
                lb=item.get("lb"),
                ub=item.get("ub"),
                fix_point=int(item.get("fix_point", 0)),
                ordered_values=item.get("ordered_values"),
            )
            if param.en_name:
                lookup[param.en_name] = param
            if param.abbreviation:
                lookup[param.abbreviation] = param
        return lookup

    def _extract_text(self, pdf_path: str) -> List[str]:
        if self.ocr_service == "google_lens":
            self.logger.info("OCR requested with service=google_lens")
            try:
                pages = self._ocr_google_lens(pdf_path)
                if self._has_usable_text(pages):
                    self.last_ocr_service_used = "google_lens"
                    self.logger.info("OCR service=google_lens completed with usable response")
                    return pages
                self.logger.warning("OCR service=google_lens returned empty response")
            except Exception as exc:
                self.logger.warning("OCR service=google_lens failed: %s", exc)

            self.logger.info("Switching OCR service to ai_models as fallback")
            pages = self._ocr_ai_models(pdf_path)
            if not self._has_usable_text(pages):
                raise RuntimeError("Fallback OCR service ai_models returned empty response")
            self.last_ocr_service_used = "ai_models"
            self.logger.info("Fallback OCR service=ai_models completed successfully")
            return pages
        if self.ocr_service == "alefba":
            self.logger.info("OCR requested with service=alefba")
            pages = self._ocr_alefba(pdf_path)
            self.last_ocr_service_used = "alefba"
            return pages
        if self.ocr_service == "local_lens_service":
            self.logger.info("OCR requested with service=local_lens_service")
            pages = self._ocr_local_lens_service(pdf_path)
            self.last_ocr_service_used = "local_lens_service"
            return pages
        if self.ocr_service == "ai_models":
            self.logger.info("OCR requested with service=ai_models")
            pages = self._ocr_ai_models(pdf_path)
            self.last_ocr_service_used = "ai_models"
            return pages
        raise ValueError(f"Unsupported OCR service: {self.ocr_service}")

    def _has_usable_text(self, pages: List[str]) -> bool:
        return any(isinstance(page, str) and page.strip() for page in pages)

    def _ocr_google_lens(self, pdf_path: str) -> List[str]:
        from io import BytesIO
        from pdf2image import convert_from_bytes
        from google.oauth2 import service_account
        from google.cloud import vision
        from google.api_core.exceptions import InternalServerError, ServiceUnavailable, DeadlineExceeded

        if not self.google_credentials_path:
            raise RuntimeError("google_credentials_path is required for google_lens")

        credentials = service_account.Credentials.from_service_account_file(
            self.google_credentials_path
        )
        client = vision.ImageAnnotatorClient(credentials=credentials)

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        images = convert_from_bytes(pdf_bytes)
        full_text: List[str] = []
        for page_index, image in enumerate(images, start=1):
            self.logger.info("Sending OCR request to google_lens page=%s", page_index)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            vision_image = vision.Image(content=image_bytes)
            response = self._vision_detect_with_retry(
                client=client,
                image=vision_image,
                page_index=page_index,
                retriable_exceptions=(InternalServerError, ServiceUnavailable, DeadlineExceeded),
            )
            full_text.append(response.full_text_annotation.text)
            self.logger.info("Received OCR response from google_lens page=%s", page_index)

        return full_text

    def _vision_detect_with_retry(
        self,
        client: Any,
        image: Any,
        page_index: int,
        retriable_exceptions: Tuple[type, ...],
        max_attempts: int = 4,
    ) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                return client.document_text_detection(image=image)
            except retriable_exceptions as exc:
                last_error = exc
                if attempt == max_attempts:
                    break
                sleep_seconds = 2 ** (attempt - 1)
                time.sleep(sleep_seconds)

        raise RuntimeError(
            f"Google Vision failed after {max_attempts} attempts on page {page_index}: {last_error}"
        ) from last_error

    def _ocr_alefba(self, pdf_path: str) -> List[str]:
        import requests

        url = self.alefba_url or os.getenv("MV0_ALEFBA_URL")
        token = self.alefba_token or os.getenv("MV0_ALEFBA_TOKEN")
        if not url or not token:
            raise RuntimeError("MV0_ALEFBA_URL and MV0_ALEFBA_TOKEN are required for alefba")

        values = {"fix_orientation": True, "type": "general"}
        headers = {"Authorization": f"Token {token}"}

        with open(pdf_path, "rb") as document:
            files = {"document": document}
            response = requests.post(
                url,
                headers=headers,
                json=values,
                files=files,
                timeout=120,
            )

        if response.status_code != 200:
            raise RuntimeError(f"Alefba error {response.status_code}: {response.text}")

        response_json = response.json()
        pages = response_json.get("pages", [])
        return [page.get("text") for page in pages if isinstance(page, dict) and page.get("text")]

    def _ocr_local_lens_service(self, pdf_path: str) -> List[str]:
        import requests

        url = self.local_lens_url or os.getenv("MV0_LOCAL_LENS_URL") or "http://vision-api.mamania.me:8000/predict/pdf/"
        self.logger.info("Sending OCR request to local_lens_service url=%s", url)

        with open(pdf_path, "rb") as document:
            files = {
                "file": (os.path.basename(pdf_path), document, "application/pdf"),
            }
            response = requests.post(url, files=files, timeout=180)

        if response.status_code != 200:
            raise RuntimeError(f"local_lens_service error {response.status_code}: {response.text}")

        response_json = response.json()
        pages = self._extract_pages_text_from_local_response(response_json)
        self.logger.info("Received OCR response from local_lens_service pages=%s", len(pages))
        if not self._has_usable_text(pages):
            raise RuntimeError("local_lens_service returned empty OCR text")
        return pages

    def _extract_pages_text_from_local_response(self, payload: Any) -> List[str]:
        collected: List[str] = []

        def _add_text(value: Any) -> None:
            if isinstance(value, str):
                txt = value.strip()
                if txt:
                    collected.append(txt)

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                # Google Vision document OCR style:
                # {"fullTextAnnotation": {"text": "..."}}
                # {"full_text_annotation": {"text": "..."}}
                for ann_key in ("fullTextAnnotation", "full_text_annotation"):
                    ann = node.get(ann_key)
                    if isinstance(ann, dict):
                        _add_text(ann.get("text"))

                # Common page-based wrappers used by OCR services.
                pages = node.get("pages")
                if isinstance(pages, list):
                    for page in pages:
                        if isinstance(page, dict):
                            _add_text(page.get("text"))
                            _walk(page)

                # Google Vision batch style: {"responses": [ ... ]}
                responses = node.get("responses")
                if isinstance(responses, list):
                    for item in responses:
                        _walk(item)

                # Generic wrappers seen in gateways/proxies.
                for wrapper in ("response", "data", "result", "results"):
                    if wrapper in node:
                        _walk(node.get(wrapper))

                # Some services return text directly.
                _add_text(node.get("text"))
                _add_text(node.get("full_text"))
                _add_text(node.get("raw_text"))
                return

            if isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(payload)

        deduped: List[str] = []
        seen: set = set()
        for text in collected:
            if text not in seen:
                deduped.append(text)
                seen.add(text)

        if deduped:
            return deduped

        if isinstance(payload, dict):
            details = f"top-level keys={list(payload.keys())}"
        else:
            details = f"payload type={type(payload).__name__}"
        raise RuntimeError(f"Unexpected local_lens_service response format ({details})")

    def _ocr_ai_models(self, pdf_path: str) -> List[str]:
        from io import BytesIO
        import requests
        from pdf2image import convert_from_bytes

        key = (
            self.ai_ocr_key
            or os.getenv("MV0_OCR_AI_KEY")
            or os.getenv("MV0_GEMINI_OCR_KEY")
            or self.openai_key
            or os.getenv("MV0_AI_KEY")
        )
        base_url = (
            self.ai_ocr_base_url
            or os.getenv("MV0_OCR_AI_BASE_URL")
            or os.getenv("MV0_GEMINI_OCR_BASE_URL")
            or self.openai_base_url
            or os.getenv("MV0_AI_BASE_URL")
        )
        model_name = (
            self.ai_ocr_model
            or os.getenv("MV0_OCR_AI_MODEL")
            or os.getenv("MV0_GEMINI_OCR_MODEL")
            or os.getenv("MV0_AI_MODEL_NAME")
        )

        if not key or not base_url or not model_name:
            raise RuntimeError(
                "Missing AI OCR config. Set MV0_OCR_AI_KEY, MV0_OCR_AI_BASE_URL, MV0_OCR_AI_MODEL "
                "(or MV0_AI_KEY, MV0_AI_BASE_URL, MV0_AI_MODEL_NAME)."
            )

        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        images = convert_from_bytes(pdf_bytes)
        full_text: List[str] = []
        for page_index, image in enumerate(images, start=1):
            self.logger.info("Sending OCR request to ai_models page=%s", page_index)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
            data_url = f"data:image/png;base64,{image_b64}"

            payload = {
                "model": model_name,
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an OCR engine. Return only extracted text.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all visible text from this image. Keep line breaks and order as much as possible. Return plain text only.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    },
                ],
            }

            page_text = self._ai_ocr_request_with_retry(
                endpoint=endpoint,
                headers=headers,
                payload=payload,
                page_index=page_index,
                requests_module=requests,
            )
            full_text.append(page_text)
            self.logger.info("Received OCR response from ai_models page=%s", page_index)

        return full_text

    def _ai_ocr_request_with_retry(
        self,
        endpoint: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        page_index: int,
        requests_module: Any,
        max_attempts: int = 4,
    ) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests_module.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"HTTP {response.status_code} on page {page_index}: {response.text}"
                    )

                response_json = response.json()
                message = (
                    response_json.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
                )
                if isinstance(message, str):
                    return message
                if isinstance(message, list):
                    parts = [part.get("text", "") for part in message if isinstance(part, dict)]
                    return "\n".join(p for p in parts if p)
                raise RuntimeError(f"Unexpected response format on page {page_index}")
            except Exception as exc:
                last_error = exc
                if attempt == max_attempts:
                    break
                sleep_seconds = 2 ** (attempt - 1)
                time.sleep(sleep_seconds)

        raise RuntimeError(
            f"AI OCR failed after {max_attempts} attempts on page {page_index}: {last_error}"
        ) from last_error

    def _text_to_json(self, text: str) -> Dict[str, Any]:
        try:
            from langchain_openai import ChatOpenAI
            try:
                from langchain.prompts import ChatPromptTemplate
            except Exception:
                from langchain_core.prompts import ChatPromptTemplate
        except Exception as exc:
            raise RuntimeError(
                "LangChain packages are required for text_to_json. "
                "Install with: pip install langchain langchain-openai. "
                f"Current python: {sys.executable}. Original import error: {exc}"
            ) from exc

        key = self.openai_key or os.getenv("MV0_AI_KEY")
        base_url = self.openai_base_url or os.getenv("MV0_AI_BASE_URL")
        model_name = self.openai_model or os.getenv("MV0_AI_MODEL_NAME")

        if not key or not base_url or not model_name:
            raise RuntimeError("MV0_AI_KEY, MV0_AI_BASE_URL, MV0_AI_MODEL_NAME are required")

        model = ChatOpenAI(
            model_name=model_name,
            base_url=base_url,
            api_key=key,
            temperature=0,
        )

        format_instructions = self._build_json_format_instructions()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.config.system_message),
                ("human", self.config.human_message),
            ]
        )
        chain = prompt | model
        response_obj = chain.invoke({
            "input_text": text,
            "format_instructions": format_instructions,
        })
        response = response_obj.content if hasattr(response_obj, "content") else str(response_obj)
        parsed = self._parse_json_response(response)

        if "PatientInfo" not in parsed or "TestResults" not in parsed:
            raise RuntimeError("LLM response missing PatientInfo or TestResults")

        return parsed

    def _build_json_format_instructions(self) -> str:
        return (
            "Return only valid JSON (no markdown, no explanation) with exactly two top-level keys: "
            '"PatientInfo" and "TestResults". '
            '"PatientInfo" must be an object. '
            '"TestResults" must be an array of objects; each object should include fields like '
            '"Test", "Result", "Unit", and "ReferenceRange" when available.'
        )

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        text = response.strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if fenced_match:
            text = fenced_match.group(1).strip()
        elif not text.startswith("{"):
            obj_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if obj_match:
                text = obj_match.group(0).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM returned invalid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError("LLM response JSON must be an object")
        return parsed

    def _map_json(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        reverse_mapping = self._get_reverse_mapping()

        mapped_test, unmapped_test = self._parse_section(
            data.get("TestResults", []), reverse_mapping, is_test=True
        )
        mapped_info, unmapped_info = self._parse_section(
            data.get("PatientInfo", {}), reverse_mapping, is_test=False
        )

        mapped_data = {**mapped_test, **mapped_info}
        unmapped_data = {**unmapped_test, **unmapped_info}
        return mapped_data, unmapped_data

    def _get_reverse_mapping(self) -> Dict[str, str]:
        reverse_map: Dict[str, str] = {}
        for key, aliases in self.config.name_mapping.items():
            if not isinstance(aliases, list):
                continue
            for alias in aliases:
                reverse_map[alias.lower()] = key
        return reverse_map

    def _parse_section(
        self, data: Any, reverse_mapping: Dict[str, str], is_test: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        mapped: Dict[str, Any] = {}
        unmapped: Dict[str, Any] = {}

        if is_test:
            for item in data:
                flat = self._extract_nested_data(item)
                name = item.get("Test", "").lower()
                base = {
                    "Result": item.get("Result"),
                    "Unit": item.get("Unit"),
                    "ReferenceRange": item.get("ReferenceRange"),
                    **{k: v for k, v in flat.items() if k not in ["Test", "Result", "Unit", "ReferenceRange"]},
                }
                target_key = reverse_mapping.get(name, item.get("Test", ""))
                if name in reverse_mapping:
                    mapped[target_key] = base
                else:
                    unmapped[target_key] = base
        else:
            flat = self._extract_nested_data(data)
            for key, value in flat.items():
                key_l = key.lower()
                target_key = reverse_mapping.get(key_l, key)
                if key_l in reverse_mapping:
                    mapped[target_key] = value
                else:
                    unmapped[target_key] = value

        return mapped, unmapped

    def _extract_nested_data(self, data: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        items: Dict[str, Any] = {}
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.update(self._extract_nested_data(value, new_key, sep))
            else:
                items[new_key] = value
        return items

    def _process_mapped(
        self, mapped_data: Dict[str, Any], unmapped_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        numeric_data, non_numeric_data, personal_info_data = self._categorize_parameters(
            mapped_data
        )

        ok_non_numeric, not_ok_non_numeric, not_found_non_numeric = self._process_non_numeric(
            non_numeric_data
        )
        ok_numeric, not_ok_numeric, not_found_numeric = self._process_numeric(numeric_data)

        ok_parameters = {**personal_info_data, **ok_non_numeric, **ok_numeric}
        not_ok_params = {
            "not_ok_non_numeric": not_ok_non_numeric,
            "not_found_non_numeric": not_found_non_numeric,
            "not_ok_numeric": not_ok_numeric,
            "not_found_numeric": not_found_numeric,
            "unmapped_data": unmapped_data,
        }

        return ok_parameters, not_ok_params

    def _categorize_parameters(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        numeric: Dict[str, Any] = {}
        non_numeric: Dict[str, Any] = {}
        personal: Dict[str, Any] = {}

        for key, value in data.items():
            param = self.param_lookup.get(key)
            if not param:
                personal[key] = value
                continue
            if param.representation_type == "numeric":
                numeric[key] = value
            else:
                non_numeric[key] = value

        return numeric, non_numeric, personal

    def _process_non_numeric(self, non_numeric_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        ok: Dict[str, Any] = {}
        not_ok: Dict[str, Any] = {}
        not_found: Dict[str, Any] = {}
        result_mapping = self.config.result_mapping

        for param, details in non_numeric_data.items():
            result = details.get("Result") if isinstance(details, dict) else details
            mapping = result_mapping.get(param)
            if not mapping:
                not_found[param] = {"group": "not_found", "value": result}
                continue

            matched = False
            for acceptable, synonyms in mapping.items():
                if result in synonyms:
                    ok[param] = {"Result": acceptable}
                    matched = True
                    break
            if not matched:
                not_ok[param] = {"group": "not_mapped", "value": result}

        return ok, not_ok, not_found

    def _process_numeric(self, numeric_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        ok: Dict[str, Any] = {}
        not_ok: Dict[str, Any] = {}
        not_found: Dict[str, Any] = {}

        for param, details in numeric_data.items():
            result = details.get("Result") if isinstance(details, dict) else details
            if result == "N/A":
                not_ok[param] = {"group": "N/A", "value": result}
                continue

            try:
                result_val = float(result)
            except (TypeError, ValueError):
                not_ok[param] = {"group": "invalid_format", "value": result}
                continue

            param_info = self.param_lookup.get(param)
            if not param_info:
                not_found[param] = {"group": "not_found", "value": result_val}
                continue

            normalized = self._normalize_numeric_value(param, result_val)
            if param_info.lb is None or param_info.ub is None:
                ok[param] = {"Result": normalized}
                continue

            if param_info.lb <= normalized <= param_info.ub:
                final_value = normalized
                if not self._decimal_within_limit(normalized, param_info.fix_point):
                    final_value = round(normalized, param_info.fix_point)
                ok[param] = {"Result": final_value}
            else:
                reason = "lower_than_min" if normalized < param_info.lb else "higher_than_max"
                not_ok[param] = {
                    "group": "out_of_range",
                    "reason": reason,
                    "value": normalized,
                }

        return ok, not_ok, not_found

    def _normalize_numeric_value(self, param: str, value: float) -> float:
        if param == "Urine Specific Gravity" and value < 30:
            return value * 1000
        if param == "Urine pH":
            return round(value)
        return value

    def _decimal_within_limit(self, value: float, limit: int) -> bool:
        str_val = f"{value:.10f}".rstrip("0").rstrip(".")
        parts = str_val.split(".")
        return len(parts) == 1 or len(parts[1]) <= limit
