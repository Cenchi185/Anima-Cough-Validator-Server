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

def update_recovery_email(user_uuid: str, recovery_email: str) -> bool:
    now = datetime.now().isoformat(timespec="seconds")

    with get_connection() as conn:
        cursor = conn.execute(
            """
            UPDATE User
            SET recovery_email = ?, updated_at = ?
            WHERE user_uuid = ?
            """,
            (recovery_email, now, user_uuid),
        )
        conn.commit()
        return cursor.rowcount > 0


def update_password(user_uuid: str, password_hash: str) -> bool:
    now = datetime.now().isoformat(timespec="seconds")

    with get_connection() as conn:
        cursor = conn.execute(
            """
            UPDATE User
            SET password_hash = ?, updated_at = ?
            WHERE user_uuid = ?
            """,
            (password_hash, now, user_uuid),
        )
        conn.commit()
        return cursor.rowcount > 0


def delete_user_account(user_uuid: str) -> bool:
    with get_connection() as conn:
        # 계정과 연결된 데이터를 함께 삭제
        conn.execute(
            "DELETE FROM MRRecord WHERE user_uuid = ?",
            (user_uuid,),
        )
        conn.execute(
            "DELETE FROM UserProfile WHERE user_uuid = ?",
            (user_uuid,),
        )
        cursor = conn.execute(
            "DELETE FROM User WHERE user_uuid = ?",
            (user_uuid,),
        )
        conn.commit()
        return cursor.rowcount > 0