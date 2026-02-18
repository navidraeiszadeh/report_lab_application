import os
from typing import Optional


class AuthenticationService:
    def __init__(self, static_api_key: Optional[str] = None):
        self.static_api_key = static_api_key or os.getenv("APP_API_KEY")

    def validate_bearer_token(self, authorization_header: Optional[str]) -> bool:
        if not self.static_api_key:
            return True
        if not authorization_header:
            return False
        prefix = "Bearer "
        if not authorization_header.startswith(prefix):
            return False
        token = authorization_header[len(prefix):].strip()
        return token == self.static_api_key
