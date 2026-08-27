# face_recognise

Django REST API for face verification: a user enrols one reference photo, then
later submits a new photo. If the two faces match to at least **80% confidence**,
a post-verification action is executed.

Recognition runs locally on CPU via [InsightFace](https://github.com/deepinsight/insightface)
(`buffalo_l` — ArcFace `w600k_r50`, 512-d embeddings) on ONNX Runtime. No image
ever leaves the machine and no third-party API is involved.

## Setup

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python manage.py migrate
./venv/bin/python manage.py createsuperuser   # optional, for /admin/
./venv/bin/python manage.py runserver
```

The model pack (~330 MB) downloads to `~/.insightface/models/` on the first
request that needs it, so that first call takes noticeably longer than the rest.
To warm it at boot instead, call `services.face_engine.get_app()` from an
`AppConfig.ready()`.

## Docker

```bash
cp .env.example .env                 # then fill in DJANGO_SECRET_KEY
docker compose up --build
```

The API is on `localhost:8000`. Two named volumes keep state out of the image:
`data` (the SQLite file and uploaded faces) and `models` (the `buffalo_l` pack,
fetched on the first request that needs it — set `FACE_WARM_MODEL=1` to pull it
at boot instead). Migrations run on every start via `docker-entrypoint.sh`;
gunicorn then serves the app, with WhiteNoise handling the admin's static files.

The image is a two-stage build on `python:3.14-slim` and runs as a non-root
user. It is large (~2 GB) — onnxruntime, scipy, scikit-image and opencv are all
runtime dependencies.

Without compose:

```bash
docker build -t face-recognise .
docker run -p 8000:8000 \
  -e DJANGO_SECRET_KEY=... -e DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1 \
  -v face-data:/data -v face-models:/home/app/.insightface \
  face-recognise
```

### Configuration

Everything below has a working default, so `manage.py runserver` still needs no
environment at all; containers should set at least the first three.

| Variable                      | Default                    | Purpose                                  |
|-------------------------------|----------------------------|------------------------------------------|
| `DJANGO_SECRET_KEY`           | insecure dev key           | Signing key — set this anywhere real     |
| `DJANGO_DEBUG`                | `True`                     | `False` in the image                     |
| `DJANGO_ALLOWED_HOSTS`        | empty                      | Comma-separated hostnames                |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | empty                      | Needed for the admin behind HTTPS        |
| `DJANGO_SQLITE_PATH`          | `BASE_DIR/db.sqlite3`      | `/data/db.sqlite3` in the image          |
| `DJANGO_MEDIA_ROOT`           | `BASE_DIR/media`           | `/data/media` in the image               |
| `DJANGO_STATIC_ROOT`          | `BASE_DIR/staticfiles`     | `collectstatic` target                   |
| `FACE_WARM_MODEL`             | `0`                        | `1` loads the model pack at boot         |
| `DJANGO_SKIP_MIGRATE`         | `0`                        | `1` skips migrations on container start  |

## Authentication

Token-based (`rest_framework.authtoken`). Send the token on every protected call:

```
Authorization: Token <token>
```

| Method | Endpoint              | Auth | Purpose                                |
|--------|-----------------------|------|----------------------------------------|
| POST   | `/api/auth/register/` | –    | Create an account, returns a token     |
| POST   | `/api/auth/login/`    | –    | Exchange credentials for a token       |
| POST   | `/api/auth/logout/`   | ✓    | Revoke the caller's token              |
| GET    | `/api/auth/me/`       | ✓    | The caller's profile                   |

```bash
curl -X POST localhost:8000/api/auth/register/ -H 'Content-Type: application/json' -d '{
  "username": "ankit", "email": "ankit@example.com",
  "password": "Str0ng-Pass-2026", "password_confirm": "Str0ng-Pass-2026"
}'
```

## Face endpoints

| Method | Endpoint               | Purpose                                        |
|--------|------------------------|------------------------------------------------|
| POST   | `/api/face/enroll/`    | Store/replace the caller's reference face      |
| POST   | `/api/face/verify/`    | Compare a new photo against the stored one     |
| GET    | `/api/face/profile/`   | Inspect the stored reference face              |
| DELETE | `/api/face/profile/`   | Delete the stored reference face               |
| GET    | `/api/face/attempts/`  | The caller's verification history              |

Both `enroll` and `verify` take `multipart/form-data` with an `image` file.
`verify` also accepts an optional `min_confidence` (0–100) to override the gate
for a single call.

```bash
curl -X POST localhost:8000/api/face/enroll/ -H "Authorization: Token $TOKEN" -F image=@reference.jpg
curl -X POST localhost:8000/api/face/verify/ -H "Authorization: Token $TOKEN" -F image=@selfie.jpg
```

A match:

```json
{
  "cosine_similarity": 0.9627,
  "confidence_percent": 100.0,
  "required_confidence_percent": 80.0,
  "above_model_threshold": true,
  "is_match": true,
  "faces_detected": 1,
  "detection_score": 0.8149,
  "face_box": [9, -2, 84, 111],
  "padded_retry": true,
  "action": {"performed": true, "name": "face_verified", "detail": "..."},
  "detail": "Face verified; action performed."
}
```

Rejections carry a stable `code`: `no_face`, `multiple_faces`, `face_too_small`,
`invalid_image`, `not_enrolled` (HTTP 409).

## The action hook

`is_match` true triggers `services/actions.py::perform_verified_action(user, result)`.
Replace its body with the real workflow (mark attendance, approve KYC, release a
document, …); whatever dict it returns is echoed to the client under `action`.

## Why 80% is not a raw cosine similarity

ArcFace produces a cosine similarity, **not** a percentage. On this project's
fixtures, genuinely different people scored between −0.08 and +0.21, while the
same person across degraded photos scored 0.82–0.99. Comparing a raw cosine
against 0.8 would work here but is far too strict in general — photos taken
years apart, at different angles, or with glasses on routinely land at 0.45–0.6
for the same person, and would be wrongly rejected.

So `confidence_percent` is a **calibrated** score: a piecewise-linear map from
cosine onto 0–100 with three anchors in `settings.py`.

```
FACE_COS_FLOOR     = 0.10   ->   0%   confidently different people
FACE_COS_THRESHOLD = 0.40   ->  50%   the model's decision boundary
FACE_COS_CEILING   = 0.70   -> 100%   unmistakably the same person
```

With those anchors, the 80% gate sits at a cosine of **0.58**. Raw
`cosine_similarity` and `above_model_threshold` are always returned alongside
it, so you can tune without guessing.

**These defaults are literature-derived, not fitted to your data.** Before going
live, calibrate against your own photos — same camera, same lighting, same
population:

```
photos/
  alice/  a1.jpg  a2.jpg  a3.jpg
  bob/    b1.jpg  b2.jpg
```

```bash
./venv/bin/python manage.py calibrate_face photos/
```

It prints the genuine/impostor cosine distributions, suggests the three anchors,
and warns when the two distributions overlap. Raising `FACE_COS_THRESHOLD`
trades false accepts for false rejects; lowering it does the reverse.

## Other tunables (`settings.py`)

| Setting                  | Default        | Effect                                          |
|--------------------------|----------------|-------------------------------------------------|
| `FACE_MODEL_NAME`        | `buffalo_l`    | `buffalo_s` is smaller/faster, less accurate    |
| `FACE_DET_SIZE`          | `(640, 640)`   | Larger finds smaller faces, costs latency       |
| `FACE_MIN_DET_SCORE`     | `0.5`          | Discard low-confidence detections               |
| `FACE_MIN_PIXELS`        | `50`           | Reject face crops smaller than this             |
| `FACE_REJECT_MULTIPLE`   | `True`         | Refuse group photos; `False` uses largest face  |
| `FACE_DETECT_PAD_RATIO`  | `0.5`          | Padding for the tight-crop retry (see below)    |
| `FACE_MIN_CONFIDENCE`    | `80.0`         | The confidence gate                             |

A tightly-cropped headshot gives the detector no margin and often yields no
detection at all. When the first pass finds nothing, the image is padded and
detection retried; `padded_retry` in the response reports when that happened.

## Tests

```bash
./venv/bin/python manage.py test                              # all 50
./venv/bin/python manage.py test users services.tests         # fast, model mocked
./venv/bin/python manage.py test services.test_engine_integration  # real model
```

The integration suite asserts on real photos that degraded images of one person
still match, and that **no** pair of different people clears the gate.

## CI

The same pipeline is defined for both hosts — [`.gitlab-ci.yml`](.gitlab-ci.yml)
and [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

| Stage       | What runs                                                                  |
|-------------|----------------------------------------------------------------------------|
| lint        | `compileall`, `manage.py check`, `makemigrations --check`                    |
| test        | the mocked suite under coverage, plus an advisory `check --deploy`           |
| build       | build the image, boot it, assert `/api/auth/me/` answers 401, push on default branch |
| integration | the real-model tests — manual on GitLab, `workflow_dispatch` on GitHub       |

Both pin the test job to the mocked labels (`users services.tests`) so CI never
downloads the 330 MB pack by accident, and both cache `~/.insightface` for the
integration job. Images push to the built-in registry of whichever host runs the
pipeline (`$CI_REGISTRY_IMAGE` / `ghcr.io/<owner>/<repo>`).

## Before production

- `DJANGO_DEBUG=False`, a real `DJANGO_SECRET_KEY`, and a real `DJANGO_ALLOWED_HOSTS`
  (the Docker image already defaults the first; the other two are yours to set).
- Serve media from S3 or similar rather than Django; reference photos are biometric
  data, so restrict access and check your retention obligations (GDPR/DPDP).
- **There is no liveness detection.** A printed photo or a phone screen held to the
  camera will pass. If verification gates anything valuable, add an anti-spoofing
  check or capture video rather than a still.
- Verification is CPU-bound (roughly 0.2–1 s per image). Under load, move it to a
  Celery task or use `onnxruntime-gpu`.
- `/api/face/verify/` is throttled to 20/min per user (`DEFAULT_THROTTLE_RATES`).
  Tighten it if verification gates something sensitive.
