from typing import Any, Dict

from database.supabase_client import get_supabase_client


class HealthDetailsRepository:
    TABLE_NAME = "health_details"

    def create_health_details(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        client = get_supabase_client()
        response = client.table(self.TABLE_NAME).insert(payload).execute()
        data = response.data or []
        if not data:
            raise RuntimeError("failed to create health_details")
        return data[0]
