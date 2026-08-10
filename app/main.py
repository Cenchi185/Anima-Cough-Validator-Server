from fastapi import FastAPI
from app.routers import (
    upload_router,
    record_router,
    profile_router,
    account_router,
)
from app.db.database import init_db

app = FastAPI()
init_db()

app.include_router(upload_router.router)
app.include_router(record_router.router)
app.include_router(profile_router.router)
app.include_router(account_router.router)

# 업로드 테스트용
@app.get("/health")
def health():
    return {"ok": True}