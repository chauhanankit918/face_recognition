#!/bin/sh
# Apply migrations, then hand off to the CMD (gunicorn by default).
set -e

if [ "${DJANGO_SKIP_MIGRATE:-0}" != "1" ]; then
  echo "==> applying migrations"
  python manage.py migrate --noinput
fi

# Optional: load the model pack now instead of on the first API call, so the
# container is not "up" until it can actually verify a face.
if [ "${FACE_WARM_MODEL:-0}" = "1" ]; then
  echo "==> warming the InsightFace model pack"
  python -c "import django; django.setup(); from services.face_engine import get_app; get_app()"
fi

exec "$@"
