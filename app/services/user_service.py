from datetime import datetime

from app.db.database import get_connection


def ensure_user_exists(
    user_uuid: str,
    device_os: str | None = None,
) -> None:
    """
    user_uuid가 User 테이블에 없으면 새로 생성.
    이미 존재하면 아무것도 하지 않음.
    """
    now = datetime.now().isoformat(timespec="seconds")

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT user_uuid
            FROM User
            WHERE user_uuid = ?
            """,
            (user_uuid,),
        )

        existing_user = cursor.fetchone()

        if existing_user is not None:
            return

        cursor.execute(
            """
            INSERT INTO User (
                user_uuid,
                device_os,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                user_uuid,
                device_os,
                now,
                now,
            ),
        )

        conn.commit()