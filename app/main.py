from fastapi import FastAPI
from app.routers import upload_router, record_router
from app.db.database import init_db

app = FastAPI()
init_db()

app.include_router(upload_router.router)
app.include_router(record_router.router)

@app.get("/health")
def health() :
    return {"ok":True}