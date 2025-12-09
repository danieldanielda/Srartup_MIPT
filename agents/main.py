import uvicorn
from fastapi import FastAPI

from src.api.v1.router import router
from src.settings.config import CrewSettings

settings = CrewSettings()

app = FastAPI(
    title="CREW",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json"
)
app.include_router(router, prefix="/api/v1/crew", tags=["Crew AI"])

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host, 
        port=int(settings.port),
        timeout_keep_alive=30,
        limit_concurrency=1000,
        log_level="info"
    )