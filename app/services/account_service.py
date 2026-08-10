from datetime import datetime

from app.db.database import get_connection


def get_user_by_user_id(user_id: str):
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *
            FROM User
            WHERE user_id = ?
            """,
            (user_id,),
        )

        return cursor.fetchone()

def get_user_by_uuid(user_uuid: str):
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *
            FROM User
            WHERE user_uuid = ?
            """,
            (user_uuid,),
        )

        return cursor.fetchone()

def link_account_to_user(
    user_uuid: str,
    user_id: str,
    password_hash: str,
    recovery_email: str | None = None,
) -> None:
    now = datetime.now().isoformat(timespec="seconds")

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE User
            SET
                user_id = ?,
                password_hash = ?,
                recovery_email = ?,
                account_linked_at = ?,
                updated_at = ?
            WHERE user_uuid = ?
            """,
            (
                user_id,
                password_hash,
                recovery_email,
                now,
                now,
                user_uuid,
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError("존재하지 않는 user_uuid입니다.")

        conn.commit()