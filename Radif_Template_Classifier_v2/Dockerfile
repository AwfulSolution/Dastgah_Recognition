# The classifier needs librosa, which needs numba, which needs llvmlite: 268 MB
# of dependencies before any code. That is why this is a container on a Python
# host rather than a serverless function -- it does not fit in one.
FROM python:3.12-slim

# libsndfile is soundfile's C library; ffmpeg lets librosa read mp3 and m4a.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies as their own layer, named rather than resolved from the project,
# so editing the classifier does not reinstall 268 MB. Installing the project
# here instead would need its sources, which would put them in this layer and
# defeat the caching. scikit-learn is absent on purpose: the pair models ship as
# coefficients and numpy evaluates them, which keeps 37 MB out of the image.
RUN pip install --no-cache-dir \
      "numpy>=1.26" "scipy>=1.11" "librosa>=0.11" "soundfile>=0.12" \
      "fastapi>=0.115" "uvicorn[standard]>=0.32" "python-multipart>=0.0.12" \
 && rm -rf /root/.cache/pip

# The package, including the notation-derived artefacts it classifies against:
# templates.json, gushehs.json and pair_models.json. /app is the working
# directory, so `dastgah` imports from here without an install step.
COPY dastgah/ dastgah/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    NUMBA_CACHE_DIR=/tmp/numba

EXPOSE 8080

# Fail the build rather than the first request if an artefact did not make it in.
RUN python -c "import dastgah.api.server; from dastgah.core.cascade import load_pair_models; \
assert load_pair_models(), 'pair_models.json missing from the image'"

# One worker: analysis is CPU-bound and already hands work to a threadpool, so a
# second worker on a shared CPU would contend rather than add throughput.
CMD ["uvicorn", "dastgah.api.server:app", "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "1", "--timeout-keep-alive", "75"]
