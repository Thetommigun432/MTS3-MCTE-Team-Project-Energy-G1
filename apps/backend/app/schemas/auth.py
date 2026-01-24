"""
Pydantic v2 schemas for authentication.
"""

from pydantic import BaseModel, ConfigDict, Field


class TokenPayloadSchema(BaseModel):
    """JWT token payload schema."""

    model_config = ConfigDict(extra="allow")

    sub: str = Field(..., description="Subject (user ID)")
    email: str | None = Field(None, description="User email")
    role: str | None = Field(None, description="User role")
    exp: int | None = Field(None, description="Expiration timestamp")
    iat: int | None = Field(None, description="Issued at timestamp")
    iss: str | None = Field(None, description="Issuer")
    aud: str | None = Field(None, description="Audience")


class CurrentUser(BaseModel):
    """Current authenticated user context."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(..., description="User ID (from JWT sub)")
    email: str | None = Field(None, description="User email")
    role: str = Field(default="user", description="User role")
