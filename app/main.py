from fastapi import FastAPI

app = FastAPI(title="ModelMux")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}