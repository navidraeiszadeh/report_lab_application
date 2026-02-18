from typing import Any, Dict, List, Optional

from database.supabase_client import get_supabase_client


class ReportRepository:
    TABLE_NAME = "reports"
    LAB_REPORTS_TABLE = "lab_reports"

    def create_report(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        client = get_supabase_client()
        response = client.table(self.TABLE_NAME).insert(payload).execute()
        data = response.data or []
        if not data:
            raise RuntimeError("failed to create report")
        return data[0]

    def list_reports(self, limit: int = 20) -> List[Dict[str, Any]]:
        client = get_supabase_client()
        response = (
            client.table(self.TABLE_NAME)
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        client = get_supabase_client()
        response = (
            client.table(self.TABLE_NAME)
            .select("*")
            .eq("id", report_id)
            .limit(1)
            .execute()
        )
        data = response.data or []
        return data[0] if data else None

    def create_lab_report(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        client = get_supabase_client()
        response = client.table(self.LAB_REPORTS_TABLE).insert(payload).execute()
        data = response.data or []
        if not data:
            raise RuntimeError("failed to create lab_report")
        return data[0]

    def list_lab_reports(self, limit: int = 20) -> List[Dict[str, Any]]:
        client = get_supabase_client()
        response = (
            client.table(self.LAB_REPORTS_TABLE)
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []

    def get_lab_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        client = get_supabase_client()
        response = (
            client.table(self.LAB_REPORTS_TABLE)
            .select("*")
            .eq("id", report_id)
            .limit(1)
            .execute()
        )
        data = response.data or []
        return data[0] if data else None
