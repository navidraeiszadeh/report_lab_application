from typing import Any, Dict

from database.supabase_client import get_supabase_client


class UserRepository:
    TABLE_NAME = "profiles"

    def username_exists(self, username: str) -> bool:
        client = get_supabase_client()
        response = (
            client.table(self.TABLE_NAME)
            .select("id", count="exact")
            .eq("username", username)
            .limit(1)
            .execute()
        )
        count = getattr(response, "count", None)
        if isinstance(count, int):
            return count > 0
        data = response.data or []
        return len(data) > 0

    def create_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        client = get_supabase_client()
        username = str(profile_data.get("username", "")).strip()
        if not username:
            raise ValueError("username is required")
        if self.username_exists(username):
            raise ValueError("username already exists")

        response = client.table(self.TABLE_NAME).insert(profile_data).execute()
        data = response.data or []
        if not data:
            raise RuntimeError("failed to create profile")
        return data[0]
