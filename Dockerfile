# syntax=docker/dockerfile:1
#
# face_recognise — Django + DRF + InsightFace (CPU, ONNX Runtime).
#
# Two stages so the compilers used to build wheels stay out of the final
# image. The image is still large (~2 GB): onnxruntime, scipy, scikit-image
# and opencv are heavy, and the runtime needs them all.

# ---------------------------------------------------------------- builder ---
FROM python:3.14-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update -qq \
 && apt-get install -yqq --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install -r /tmp/requirements.txt

# ---------------------------------------------------------------- runtime ---
FROM python:3.14-slim AS runtime

# opencv links against libGL/glib even when it is only imported.
RUN apt-get update -qq \
 && apt-get install -yqq --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DJANGO_SETTINGS_MODULE=face_recognise.settings \
    DJANGO_DEBUG=False \
    DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1 \
    DJANGO_SQLITE_PATH=/data/db.sqlite3 \
    DJANGO_MEDIA_ROOT=/data/media \
    DJANGO_STATIC_ROOT=/app/staticfiles \
    HOME=/home/app

COPY --from=builder /opt/venv /opt/venv

# Non-root. /data holds the SQLite file and uploaded faces; ~/.insightface
# holds the ~330 MB model pack, fetched on the first request that needs it.
RUN useradd --create-home --home-dir /home/app --uid 1000 app \
 && mkdir -p /app /data/media /home/app/.insightface \
 && chown -R app:app /app /data /home/app

WORKDIR /app
COPY --chown=app:app . /app
COPY --chown=app:app docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

USER app

# Baked in at build time so start-up is just migrate + serve.
RUN python manage.py collectstatic --noinput

VOLUME ["/data", "/home/app/.insightface"]
EXPOSE 8000

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["gunicorn", "face_recognise.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-"]
