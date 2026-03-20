from app.services.model_loader import load_assets
from app.routers.upload_router import router as upload_router

from fastapi import FastAPI

app = FastAPI()
app.include_router(upload_router)

# 서버 시작(startup)됬을때 ***딱 한번만*** 호출됨
@app.on_event("startup")
def on_startup() -> None:
    load_assets()