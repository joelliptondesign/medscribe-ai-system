from fastapi import FastAPI

from service.api import router


app = FastAPI(title="med_scribe")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "med_scribe",
        "status": "ok",
    }


app.include_router(router)
