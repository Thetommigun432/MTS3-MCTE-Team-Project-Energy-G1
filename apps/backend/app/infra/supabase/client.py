"""
Supabase client wrapper for metadata queries.
Used for authorization data lookups, NOT per-request JWT verification.
"""

from typing import Any

from supabase import create_client, Client

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class SupabaseClient:
    """Supabase client wrapper for metadata queries."""

    def __init__(self) -> None:
        self._client: Client | None = None

    def connect(self) -> None:
        """Initialize the Supabase client."""
        settings = get_settings()

        # Prefer publishable key, fallback handled in config or here if needed
        key = settings.supabase_publishable_key or settings.supabase_anon_key

        if not settings.supabase_url or not key:
            logger.warning("Supabase credentials not configured")
            return

        self._client = create_client(
            settings.supabase_url,
            key,
        )
        logger.info("Supabase client connected")

    @property
    def client(self) -> Client | None:
        """Get the underlying Supabase client."""
        return self._client

    async def get_user_buildings(self, user_id: str) -> list[str]:
        """
        Get building IDs that a user owns.

        Returns:
            List of building UUIDs
        """
        if not self._client:
            logger.warning("Supabase client not connected")
            return []

        try:
            response = (
                self._client.table("buildings")
                .select("id")
                .eq("user_id", user_id)
                .execute()
            )
            return [row["id"] for row in response.data]
        except Exception as e:
            logger.error("Failed to get user buildings", extra={"user_id": user_id, "error": str(e)})
            return []

    async def get_building_appliances(self, building_id: str) -> list[str]:
        """
        Get appliance IDs associated with a building.

        Returns:
            List of org_appliance_ids linked to the building
        """
        if not self._client:
            logger.warning("Supabase client not connected")
            return []

        try:
            response = (
                self._client.table("building_appliances")
                .select("org_appliance_id")
                .eq("building_id", building_id)
                .eq("is_enabled", True)
                .execute()
            )
            return [row["org_appliance_id"] for row in response.data]
        except Exception as e:
            logger.error(
                "Failed to get building appliances",
                extra={"building_id": building_id, "error": str(e)},
            )
            return []

    async def get_user_role(self, user_id: str) -> str | None:
        """
        Get user role from profiles table.
        WARNING: The 'role' column might not exist in the 'profiles' table in all environments.
        Prefer obtaining role from JWT claims where possible.

        Returns:
            Role string or None if not found
        """
        if not self._client:
            return None

        try:
            # We select * to see what we get, or specific columns if we were sure.
            # But here we try 'role' specifically. If it fails, we catch it.
            response = (
                self._client.table("profiles")
                .select("role")
                .eq("id", user_id)
                .single()
                .execute()
            )
            return response.data.get("role") if response.data else None
        except Exception:
            # Column might not exist or user not found
            return None

    async def get_org_appliance_by_slug(self, slug: str) -> dict[str, Any] | None:
        """
        Get org appliance by slug (for mapping appliance_id to org_appliance_id).

        Returns:
            Org appliance data or None
        """
        if not self._client:
            return None

        try:
            response = (
                self._client.table("org_appliances")
                .select("*")
                .eq("slug", slug)
                .single()
                .execute()
            )
            return response.data
        except Exception:
            return None


# Global client instance
_supabase_client: SupabaseClient | None = None


def get_supabase_client() -> SupabaseClient:
    """Get the global Supabase client instance."""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = SupabaseClient()
    return _supabase_client


def init_supabase_client() -> SupabaseClient:
    """Initialize the global Supabase client."""
    client = get_supabase_client()
    client.connect()
    return client
