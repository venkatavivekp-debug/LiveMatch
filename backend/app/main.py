import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.catalog import router as catalog_router
from app.api.routes.insights import router as insights_router
from app.api.routes.prediction import router as prediction_router
from app.api.routes.research import router as research_router
from app.core.config import get_settings
from app.core.runtime_checks import run_startup_checks
from app.db import models  # noqa: F401
from app.db.init_db import init_db
from app.services.prediction_service import PredictionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(title=settings.project_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    run_startup_checks()
    init_db()
    mode = PredictionService.runtime_mode()
    logger.info("Prediction service initialized in %s mode.", mode)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(catalog_router)
app.include_router(prediction_router)
app.include_router(insights_router)
app.include_router(research_router)
