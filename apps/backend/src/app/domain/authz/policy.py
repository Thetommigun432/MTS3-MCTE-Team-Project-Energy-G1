"""
Authorization policy and permission checks.
Implements RBAC + resource-based access control.
"""

import time
from typing import Any

from app.core.config import get_settings
from app.core.errors import AuthenticationError, AuthorizationError, ErrorCode
from app.core.logging import get_logger
from app.core.security import TokenPayload
from app.core.telemetry import AUTHZ_CHECK_LATENCY
from app.infra.supabase.cache import PermissionGraph, get_authz_cache
from app.infra.supabase.client import get_supabase_client

logger = get_logger(__name__)

# Demo/development buildings that are accessible without Supabase
DEMO_BUILDINGS = frozenset({
    "demo-residential-001",
    "building-1",
    "building_1",  # Alternative format
})


class AuthzPolicy:
    """Authorization policy checker."""

    ADMIN_ROLES = frozenset({"admin", "super_admin"})

    async def get_permission_graph(self, user_id: str, jwt_role: str | None = None) -> PermissionGraph:
        """
        Get or build the permission graph for a user.

        Uses cache with 60s TTL to reduce Supabase calls.
        """
        cache = get_authz_cache()

        # Check cache first
        cached = await cache.get(user_id)
        if cached:
            return cached

        # Build permission graph from Supabase
        supabase = get_supabase_client()

        # Get user's buildings
        buildings = await supabase.get_user_buildings(user_id)

        # Get appliances for each building
        building_appliances: dict[str, list[str]] = {}
        for building_id in buildings:
            appliances = await supabase.get_building_appliances(building_id)
            building_appliances[building_id] = appliances

        # Get role (prefer JWT claim, fall back to database)
        role = jwt_role
        if not role:
            role = await supabase.get_user_role(user_id) or "user"

        graph = PermissionGraph(
            buildings=buildings,
            building_appliances=building_appliances,
            role=role,
        )

        # Cache the result
        await cache.set(user_id, graph)

        return graph

    async def check_building_access(
        self,
        token: TokenPayload,
        building_id: str,
    ) -> bool:
        """
        Check if user has access to a building.

        Access is granted if:
        - User has admin role
        - Building is in the user's permission graph (owned, member, or demo)
        - Building is a known demo building and we're in dev mode
        """
        start_time = time.time()
        settings = get_settings()

        try:
            # In dev/test mode, allow access to demo buildings for any authenticated user
            if settings.env in ("dev", "test", "local") and building_id in DEMO_BUILDINGS:
                logger.debug(
                    "Allowing demo building access in dev mode",
                    extra={"building_id": building_id, "user_id": token.user_id}
                )
                return True

            graph = await self.get_permission_graph(token.user_id, token.role)

            # Admins have universal access
            if graph.role in self.ADMIN_ROLES:
                return True

            # Check building in permission graph
            if building_id in graph.buildings:
                return True
            
            # If Supabase returned no buildings but we have demo buildings configured,
            # allow access to demo buildings for authenticated users
            if len(graph.buildings) == 0 and building_id in DEMO_BUILDINGS:
                logger.info(
                    "Allowing demo building access (Supabase returned no buildings)",
                    extra={"building_id": building_id, "user_id": token.user_id}
                )
                return True

            return False

        finally:
            duration = time.time() - start_time
            AUTHZ_CHECK_LATENCY.labels(check_type="building").observe(duration)

    async def check_appliance_access(
        self,
        token: TokenPayload,
        building_id: str,
        appliance_id: str,
    ) -> bool:
        """
        Check if user has access to an appliance within a building.

        Requires:
        1. User has building access
        2. Appliance is associated with the building
        """
        start_time = time.time()

        try:
            graph = await self.get_permission_graph(token.user_id, token.role)

            # Admins have universal access
            if graph.role in self.ADMIN_ROLES:
                return True

            # Check building ownership
            if building_id not in graph.buildings:
                return False

            # Check appliance association
            building_appliances = graph.building_appliances.get(building_id, [])
            return appliance_id in building_appliances

        finally:
            duration = time.time() - start_time
            AUTHZ_CHECK_LATENCY.labels(check_type="appliance").observe(duration)

    async def check_admin_role(self, token: TokenPayload) -> bool:
        """Check if user has admin role."""
        graph = await self.get_permission_graph(token.user_id, token.role)
        return graph.role in self.ADMIN_ROLES

    def require_building_access(
        self,
        token: TokenPayload,
        building_id: str,
    ) -> None:
        """
        Require building access, raising AuthorizationError if denied.

        Note: This is a sync wrapper, use check_building_access for async.
        """
        # This needs to be called from async context
        raise NotImplementedError("Use check_building_access in async context")


async def require_building_access(
    token: TokenPayload,
    building_id: str,
) -> None:
    """
    Require building access, raising AuthorizationError if denied.
    """
    policy = AuthzPolicy()
    has_access = await policy.check_building_access(token, building_id)

    if not has_access:
        raise AuthorizationError(
            code=ErrorCode.AUTHZ_BUILDING_ACCESS_DENIED,
            message=f"Access denied to building: {building_id}",
            details={"building_id": building_id},
        )


async def require_building_access_or_demo(
    token: TokenPayload | None,
    building_id: str,
) -> None:
    """
    Require building access OR allow access to demo buildings without auth.
    
    This allows unauthenticated access to demo buildings (building-1, demo-residential-001)
    while still requiring authentication for other buildings.
    """
    # If it's a demo building, allow access without authentication
    if building_id in DEMO_BUILDINGS:
        logger.debug(
            "Allowing unauthenticated access to demo building",
            extra={"building_id": building_id}
        )
        return
    
    # For non-demo buildings, require authentication
    if token is None:
        raise AuthenticationError(
            code=ErrorCode.AUTH_MISSING_TOKEN,
            message="Authentication required for this building",
        )
    
    # Check building access via policy
    policy = AuthzPolicy()
    has_access = await policy.check_building_access(token, building_id)

    if not has_access:
        raise AuthorizationError(
            code=ErrorCode.AUTHZ_BUILDING_ACCESS_DENIED,
            message=f"Access denied to building: {building_id}",
            details={"building_id": building_id},
        )


async def require_appliance_access(
    token: TokenPayload,
    building_id: str,
    appliance_id: str,
) -> None:
    """
    Require appliance access, raising AuthorizationError if denied.
    """
    policy = AuthzPolicy()
    has_access = await policy.check_appliance_access(token, building_id, appliance_id)

    if not has_access:
        raise AuthorizationError(
            code=ErrorCode.AUTHZ_APPLIANCE_ACCESS_DENIED,
            message=f"Access denied to appliance: {appliance_id}",
            details={"building_id": building_id, "appliance_id": appliance_id},
        )


async def require_admin_role(token: TokenPayload) -> None:
    """
    Require admin role, raising AuthorizationError if not admin.
    """
    policy = AuthzPolicy()
    is_admin = await policy.check_admin_role(token)

    if not is_admin:
        raise AuthorizationError(
            code=ErrorCode.AUTHZ_INSUFFICIENT_ROLE,
            message="Admin role required",
        )
