from fastapi import Header, HTTPException, status

from .config import settings


def require_admin(x_admin_password: str | None = Header(default=None)) -> None:
    if not x_admin_password or x_admin_password != settings.admin_password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="admin password required")
