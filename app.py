"""Vercel ASGI entrypoint for CerBerus Pipeline Builder.

This entrypoint is intentionally thin: it proves the repository can be
built and served by Vercel without starting the long-running Gradio server.
The existing Pipeline Builder code remains unchanged.
"""

from fastapi import FastAPI

app = FastAPI(
    title="CerBerus Pipeline Builder",
    version="1.1",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/")
async def root():
    return {
        "service": "CerBerus Pipeline Builder",
        "status": "online",
        "runtime": "vercel-python",
    }


@app.get("/health")
async def health():
    return {"ok": True}
