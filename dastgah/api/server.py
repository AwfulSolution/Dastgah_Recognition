"""HTTP API over the classifier, for the web front end.

Analysis is CPU-bound and takes several seconds per minute of audio, so upload
handling runs the work in a thread rather than blocking the event loop.
"""

from __future__ import annotations

import logging
import tempfile
import warnings
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from dastgah.core.analyze import DEFAULT_GUSHEH_PATH, DEFAULT_TEMPLATE_PATH, analyze
from dastgah.core.musicxml import scale_to_musicxml
from dastgah.radif.gusheh import load_gusheh_templates
from dastgah.radif.templates import load_templates
from dastgah.theory import MODAL_CLASSES

logger = logging.getLogger(__name__)

#: Uploads above this are rejected; a few minutes of lossless audio fits.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

SUPPORTED_SUFFIXES = {".wav", ".flac", ".aiff", ".aif", ".mp3", ".m4a", ".ogg", ".opus"}

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load templates once at startup so the first upload is not slower."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    templates()
    gusheh_templates()
    yield


app = FastAPI(
    title="Dastgah Classifier",
    description="Modal classification of Persian classical music.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_templates = None
_gushehs = None


def templates():
    global _templates
    if _templates is None:
        _templates = load_templates(DEFAULT_TEMPLATE_PATH)
    return _templates


def gusheh_templates():
    global _gushehs
    if _gushehs is None:
        _gushehs = load_gusheh_templates(DEFAULT_GUSHEH_PATH)
    return _gushehs


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "templates": len(templates())}


@app.get("/api/modes")
def modes() -> dict:
    """The 13 modal classes and the template statistics behind each."""
    loaded = templates()
    return {
        "modes": [
            {
                "key": m.key,
                "name": m.name,
                "display": m.display,
                "persian": m.persian,
                "kind": m.kind,
                "parent": m.parent,
                "tonic": loaded[m.key].tonic_pc if m.key in loaded else None,
                "shahed_interval": loaded[m.key].shahed_interval if m.key in loaded else None,
                "n_gushehs": loaded[m.key].n_gushehs if m.key in loaded else 0,
                "tonic_confidence": loaded[m.key].tonic_confidence if m.key in loaded else 0.0,
            }
            for m in MODAL_CLASSES
        ]
    }


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...)) -> dict:
    """Analyse an uploaded recording and return its modal classification."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported format {suffix or '(none)'}; "
            f"expected one of {', '.join(sorted(SUPPORTED_SUFFIXES))}",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file is {len(payload) / 1e6:.0f} MB; limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as handle:
        handle.write(payload)
        handle.flush()
        try:
            result = await run_in_threadpool(
                analyze,
                Path(handle.name),
                templates=templates(),
                gusheh_templates=gusheh_templates(),
            )
        except ValueError as exc:
            # Report the name the user uploaded, not the temporary file's.
            detail = str(exc).replace(Path(handle.name).name, file.filename or "upload")
            raise HTTPException(status_code=422, detail=detail) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("analysis failed for %s", file.filename)
            raise HTTPException(status_code=500, detail=f"analysis failed: {exc}") from exc

    payload = result.to_dict()
    payload["source"] = file.filename
    # Served alongside the analysis so the front end can offer a download
    # without re-uploading the audio; a scale is only a couple of kilobytes.
    payload["musicxml"] = scale_to_musicxml(result)
    return payload
