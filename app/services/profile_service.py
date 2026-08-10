import json
from datetime import datetime

from app.db.database import get_connection


def save_or_update_profile(profile: dict) -> None:
    now = datetime.now().isoformat(timespec="seconds")

    symptoms = profile.get("symptoms")
    if symptoms is not None:
        symptoms = json.dumps(symptoms, ensure_ascii=False)

    respiratory_conditions = profile.get("respiratory_conditions")
    if respiratory_conditions is not None:
        respiratory_conditions = json.dumps(
            respiratory_conditions,
            ensure_ascii=False,
        )

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO UserProfile (
                user_uuid,
                age,
                gender,
                country,
                region,
                language,
                visit_purpose,
                first_visit,
                smoking,
                allergy,
                allergy_detail,
                dust_exposure,
                has_pet,
                living_environment,
                symptoms,
                symptom_duration,
                respiratory_conditions,
                recent_infection,
                medication,
                note,
                consented_at,
                created_at,
                updated_at
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?
            )
            ON CONFLICT(user_uuid) DO UPDATE SET
                age = excluded.age,
                gender = excluded.gender,
                country = excluded.country,
                region = excluded.region,
                language = excluded.language,
                visit_purpose = excluded.visit_purpose,
                first_visit = excluded.first_visit,
                smoking = excluded.smoking,
                allergy = excluded.allergy,
                allergy_detail = excluded.allergy_detail,
                dust_exposure = excluded.dust_exposure,
                has_pet = excluded.has_pet,
                living_environment = excluded.living_environment,
                symptoms = excluded.symptoms,
                symptom_duration = excluded.symptom_duration,
                respiratory_conditions = excluded.respiratory_conditions,
                recent_infection = excluded.recent_infection,
                medication = excluded.medication,
                note = excluded.note,
                consented_at = excluded.consented_at,
                updated_at = excluded.updated_at
            """,
            (
                profile["user_uuid"],
                profile.get("age"),
                profile.get("gender"),
                profile.get("country"),
                profile.get("region"),
                profile.get("language"),
                profile.get("visit_purpose"),
                profile.get("first_visit"),
                profile.get("smoking"),
                profile.get("allergy"),
                profile.get("allergy_detail"),
                profile.get("dust_exposure"),
                profile.get("has_pet"),
                profile.get("living_environment"),
                symptoms,
                profile.get("symptom_duration"),
                respiratory_conditions,
                profile.get("recent_infection"),
                profile.get("medication"),
                profile.get("note"),
                profile.get("consented_at"),
                now,
                now,
            ),
        )

        conn.commit()