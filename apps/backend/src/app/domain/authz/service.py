"""
Authorization service aggregating policy operations.
"""

from app.core.security import TokenPayload
from app.domain.authz.policy import (
    AuthzPolicy,
    require_admin_role,
    require_appliance_access,
    require_building_access,
)
from app.infra.supabase.cache import get_authz_cache


class AuthzService:
    """Authorization service for access control."""

    def __init__(self) -> None:
        self._policy = AuthzPolicy()

    async def check_building_access(
        self,
        token: TokenPayload,
        building_id: str,
    ) -> bool:
        """Check if user can access a building."""
        return await self._policy.check_building_access(token, building_id)

    async def check_appliance_access(
        self,
        token: TokenPayload,
        building_id: str,
        appliance_id: str,
    ) -> bool:
        """Check if user can access an appliance."""
        return await self._policy.check_appliance_access(token, building_id, appliance_id)

    async def check_admin(self, token: TokenPayload) -> bool:
        """Check if user is admin."""
        return await self._policy.check_admin_role(token)

    async def invalidate_user_cache(self, user_id: str) -> bool:
        """Invalidate cache for a specific user."""
        cache = get_authz_cache()
        return await cache.invalidate(user_id)

    async def invalidate_all_cache(self) -> int:
        """Invalidate all cached entries."""
        cache = get_authz_cache()
        return await cache.invalidate_all()

    async def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        cache = get_authz_cache()
        return await cache.get_stats()


# Module exports
__all__ = [
    "AuthzService",
    "AuthzPolicy",
    "require_building_access",
    "require_appliance_access",
    "require_admin_role",
]
