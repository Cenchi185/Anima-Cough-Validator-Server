from fastapi import FastAPI
from app.routers import upload_router

app = FastAPI()
app.include_router(upload_router.router)

@app.get("/health")
def health() :
    return {"ok":True}