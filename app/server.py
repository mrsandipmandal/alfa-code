"""Alfa-Code web server: serves the dashboard page + generation API.

Run locally:  python app/server.py   ->  http://127.0.0.1:8000
Env: ALFA_MODEL / ALFA_TOKENIZER (see app/model_backend.py),
     ALFA_PORT (default 8000).

Endpoints:
  GET  /            working app page (app/static/alfa.html)
  GET  /design      design showcase page (app/static/alfa-design.html)
  GET  /api/status  {mode, device, reason} for the engine badge
  POST /api/generate JSON {prompt, max_tokens?, temperature?}
       or multipart form (prompt, max_tokens, file)
       -> {status, markdown, engine}
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse

from model_backend import generate, get_backend

STATIC = Path(__file__).resolve().parent / "static" / "alfa.html"

app = FastAPI(title="Alfa-Code")


@app.get("/")
def index():
    return FileResponse(str(STATIC))


@app.get("/design")
def design():
    return FileResponse(str(STATIC.parent / "alfa-design.html"))


@app.get("/api/status")
def status():
    b = get_backend()
    return {"mode": b["mode"], "device": b["device"], "reason": b["reason"]}


def _run(prompt: str, path=None, max_tokens: int = 256, temperature: float = 0.6):
    final_md, final_status = "", "[Done]"
    for final_status, final_md in generate(prompt, None, path, None,
                                           max_new_tokens=max_tokens,
                                           temperature=temperature):
        pass
    return {"status": final_status, "markdown": final_md,
            "engine": get_backend()["mode"]}


@app.post("/api/generate")
async def generate_ep(request: Request):
    ctype = request.headers.get("content-type", "")
    path = None
    try:
        if ctype.startswith("multipart/"):
            form = await request.form()
            prompt = str(form.get("prompt", ""))
            max_tokens = int(form.get("max_tokens") or 256)
            f = form.get("file")
            if f is not None and hasattr(f, "read"):
                suffix = Path(getattr(f, "filename", "") or "upload").suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(await f.read())
                    path = tmp.name
            return _run(prompt, path, max_tokens)
        payload = await request.json()
        return _run(str(payload.get("prompt", "")),
                    max_tokens=int(payload.get("max_tokens", 256)),
                    temperature=float(payload.get("temperature", 0.6)))
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("ALFA_PORT", "8000")))
