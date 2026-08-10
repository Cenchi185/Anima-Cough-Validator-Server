import sqlite3
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import HashingError
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, EmailStr

from app.services.account_service import (
    get_user_by_user_id,
    get_user_by_uuid,
    link_account_to_user,
)
from app.services.user_service import ensure_user_exists

router = APIRouter(prefix="/account", tags=["account"])

password_hasher = PasswordHasher()

class RegisterRequest(BaseModel):
    user_uuid: str = Field(min_length=1, max_length=100)
    user_id: str = Field(min_length=4, max_length=50)
    password: str = Field(min_length=8, max_length=128)
    recovery_email: Optional[EmailStr] = None

class LoginRequest(BaseModel):
    user_id: str = Field(min_length=4, max_length=50)
    password: str = Field(min_length=8, max_length=128)

@router.post("/register")
def register_account(payload: RegisterRequest):
    existing_user = get_user_by_user_id(payload.user_id)

    if existing_user is not None:
        raise HTTPException(
            status_code=409,
            detail="이미 사용 중인 user_id입니다.",
        )

    ensure_user_exists(
        user_uuid=payload.user_uuid,
        device_os="WEB",
    )

    current_user = get_user_by_uuid(payload.user_uuid)

    if current_user is not None and current_user["user_id"] is not None:
        raise HTTPException(
            status_code=409,
            detail="이미 계정이 연결된 사용자입니다.",
        )

    try:
        password_hash = password_hasher.hash(payload.password)

    except HashingError:
        raise HTTPException(
            status_code=500,
            detail="비밀번호 처리 중 오류가 발생했습니다.",
        )

    try:
        link_account_to_user(
            user_uuid=payload.user_uuid,
            user_id=payload.user_id,
            password_hash=password_hash,
            recovery_email=payload.recovery_email,
        )

    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409,
            detail="이미 사용중인 계정 정보입니다.",
        )

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )

    return {
        "ok": True,
        "user_uuid": payload.user_uuid,
        "user_id": payload.user_id,
    }

@router.post("/login")
def login_account(payload: LoginRequest):
    user = get_user_by_user_id(payload.user_id)

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="아이디 또는 비밀번호가 올바르지 않습니다.",
        )

    password_hash = user["password_hash"]

    if password_hash is None:
        raise HTTPException(
            status_code=401,
            detail="아이디 또는 비밀번호가 올바르지 않습니다.",
        )

    try:
        password_hasher.verify(
            password_hash,
            payload.password,
        )

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="아이디 또는 비밀번호가 올바르지 않습니다.",
        )

    return {
        "ok": True,
        "user_uuid": user["user_uuid"],
        "user_id": user["user_id"],
    }