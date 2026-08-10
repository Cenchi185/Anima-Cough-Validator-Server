import sqlite3
from datetime import datetime
from typing import Optional, List, Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.profile_service import save_or_update_profile
from app.services.user_service import ensure_user_exists

router = APIRouter(prefix="/profile", tags=["profile"])


class UserProfileRequest(BaseModel):
    user_uuid: str

    age: Optional[int] = Field(default=None, ge=1, le=120)
    gender: Optional[Literal["male", "female", "other"]] = None
    country: Optional[Literal["kr","other"]] = None
    region: Optional[str] = None
    language: Optional[Literal["ko","en"]] = None
    visit_purpose: Optional[Literal["self_check","research","curiosity"]] = None
    first_visit: Optional[bool] = None

    smoking: Optional[Literal["current", "past", "never"]] = None
    allergy: Optional[bool] = None
    allergy_detail: Optional[str] = None
    dust_exposure: Optional[bool] = None
    has_pet: Optional[bool] = None
    living_environment: Optional[Literal["urban","suburban","rural","industrial","other"]] = None

    symptoms: Optional[List[Literal["cough","sputum","runny_nose","sore_throat","dyspnea","fever"]]] = None
    symptom_duration: Optional[Literal["asthma","copd","rhinitis","sinusitis","bronchitis","tuberculosis"]] = None
    respiratory_conditions: Optional[List[str]] = None
    recent_infection: Optional[bool] = None
    medication: Optional[str] = None
    note: Optional[str] = None

    consented_at: Optional[datetime] = None


@router.post("")
def save_profile(payload: UserProfileRequest):
    try:
        ensure_user_exists(
            user_uuid=payload.user_uuid,
            device_os="WEB",
        )

        profile_data = payload.model_dump()

        if profile_data.get("consented_at") is not None:
            profile_data["consented_at"] = profile_data["consented_at"].isoformat()

        # SQLite에는 bool 대신 0/1로 저장
        for field_name in [
            "first_visit",
            "allergy",
            "dust_exposure",
            "has_pet",
            "recent_infection",
        ]:
            value = profile_data.get(field_name)

            if value is not None:
                profile_data[field_name] = 1 if value else 0

        save_or_update_profile(profile_data)

        return {
            "ok": True,
            "user_uuid": payload.user_uuid,
        }

    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=400,
            detail="프로필 데이터를 저장할 수 없습니다.",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="프로필 저장 중 서버 오류가 발생했습니다.",
        )