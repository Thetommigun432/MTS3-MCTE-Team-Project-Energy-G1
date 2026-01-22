"""
Admin API endpoints.
"""

from typing import Any

from fastapi import APIRouter, Body, Depends

from app.api.deps import CurrentUserDep, RequestIdDep, require_admin_token
from app.domain.authz import AuthzService, require_admin_role
from app.domain.inference import get_inference_service

router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
    dependencies=[Depends(require_admin_token)],
)


@router.post("/reload-models")
async def reload_models(
    current_user: CurrentUserDep,
    request_id: RequestIdDep,
) -> dict[str, Any]:
    """
    Reload model registry and clear model cache.

    Requires admin role.
    """
    await require_admin_role(current_user)

    service = get_inference_service()
    result = await service.reload_models()

    return {
        "status": "ok",
        "request_id": request_id,
        **result,
    }


@router.post("/cache/invalidate")
async def invalidate_cache(
    current_user: CurrentUserDep,
    request_id: RequestIdDep,
    user_id: str | None = Body(None, embed=True),
    all_users: bool = Body(False, embed=True, alias="all"),
) -> dict[str, Any]:
    """
    Invalidate authorization cache.

    Body options:
    - {"user_id": "..."} - Invalidate specific user
    - {"all": true} - Invalidate all users

    Requires admin role.
    """
    await require_admin_role(current_user)

    authz = AuthzService()

    if all_users:
        count = await authz.invalidate_all_cache()
        return {
            "status": "ok",
            "request_id": request_id,
            "invalidated": "all",
            "count": count,
        }
    elif user_id:
        success = await authz.invalidate_user_cache(user_id)
        return {
            "status": "ok",
            "request_id": request_id,
            "invalidated": user_id,
            "found": success,
        }
    else:
        return {
            "status": "error",
            "request_id": request_id,
            "message": "Specify 'user_id' or 'all: true'",
        }


@router.get("/cache/stats")
async def cache_stats(
    current_user: CurrentUserDep,
    request_id: RequestIdDep,
) -> dict[str, Any]:
    """
    Get authorization cache statistics.

    Requires admin role.
    """
    await require_admin_role(current_user)

    authz = AuthzService()
    stats = await authz.get_cache_stats()

    return {
        "status": "ok",
        "request_id": request_id,
        "cache": stats,
    }
