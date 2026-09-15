from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone
import mimetypes
import oci
import os

BUCKET_NAME = "anima-audio-storage"

@lru_cache(maxsize=1)
def get_object_storage_client():
    auth_mode = os.getenv("OCI_AUTH_MODE", "instance_principal")

    if auth_mode == "config":
        config = oci.config.from_file()
        client = oci.object_storage.ObjectStorageClient(config)
    else:
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        client = oci.object_storage.ObjectStorageClient(
            config={},
            signer=signer,
        )

    namespace = client.get_namespace().data
    return client, namespace


def build_audio_object_name(
    user_uuid: str,
    record_uuid: str,
    suffix: str = ".wav",
) -> str:
    now = datetime.now(timezone.utc)
    safe_user_uuid = user_uuid or "local"

    return (
        f"audio/"
        f"user={safe_user_uuid}/"
        f"year={now.year}/"
        f"month={now.month:02}/"
        f"{record_uuid}{suffix}"
    )


def upload_file_to_object_storage(
    file_path: str | Path,
    object_name: str,
    content_type: str | None = None,
) -> dict:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"upload target not found: {path}")

    client, namespace = get_object_storage_client()

    if content_type is None:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

    with path.open("rb") as f:
        response = client.put_object(
            namespace_name=namespace,
            bucket_name=BUCKET_NAME,
            object_name=object_name,
            put_object_body=f,
            content_type=content_type,
        )

    return {
        "namespace": namespace,
        "bucket": BUCKET_NAME,
        "object_name": object_name,
        "etag": response.headers.get("etag"),
    }