"""Face detection / embedding / comparison built on InsightFace (ONNX Runtime, CPU).

The model pack is loaded once per process, lazily, on first use.
"""
import threading

import cv2
import numpy as np
from django.conf import settings

_app = None
_lock = threading.Lock()


class FaceError(Exception):
    """Raised when an image cannot be turned into a usable face embedding."""

    def __init__(self, message, code):
        super().__init__(message)
        self.message = message
        self.code = code


def get_app():
    """Return the shared FaceAnalysis instance, loading the models on first call."""
    global _app
    if _app is None:
        with _lock:
            if _app is None:
                from insightface.app import FaceAnalysis

                app = FaceAnalysis(
                    name=settings.FACE_MODEL_NAME,
                    # Skip the landmark/gender-age models: verification only needs
                    # a bounding box plus the 512-d recognition embedding.
                    allowed_modules=['detection', 'recognition'],
                    providers=['CPUExecutionProvider'],
                )
                app.prepare(ctx_id=-1, det_size=settings.FACE_DET_SIZE)
                _app = app
    return _app


def decode_image(raw_bytes):
    """Decode uploaded bytes into a BGR numpy array."""
    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise FaceError('Could not decode the image file.', 'invalid_image')
    return image


def detect_faces(image):
    """Return detected faces sorted largest-first, plus the pad offset applied.

    A tightly-cropped headshot gives the detector no surrounding context and
    often yields nothing, so an empty first pass is retried on a padded copy.
    """
    faces = _detect(image)
    if faces:
        return faces, 0

    pad = int(min(image.shape[:2]) * settings.FACE_DETECT_PAD_RATIO)
    if pad <= 0:
        return [], 0
    padded = cv2.copyMakeBorder(
        image, pad, pad, pad, pad, cv2.BORDER_REPLICATE
    )
    return _detect(padded), pad


def _detect(image):
    faces = [
        f for f in get_app().get(image)
        if f.det_score >= settings.FACE_MIN_DET_SCORE
    ]
    faces.sort(key=lambda f: _box_area(f.bbox), reverse=True)
    return faces


def _box_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def get_embedding(raw_bytes):
    """Extract a unit-norm 512-d embedding for the most prominent face.

    Returns (embedding, info) where info carries detection metadata worth
    surfacing to the caller.
    """
    image = decode_image(raw_bytes)
    faces, pad = detect_faces(image)

    if not faces:
        raise FaceError('No face detected in the image.', 'no_face')

    if len(faces) > 1 and settings.FACE_REJECT_MULTIPLE:
        raise FaceError(
            f'{len(faces)} faces detected; the image must contain exactly one face.',
            'multiple_faces',
        )

    face = faces[0]
    x1, y1, x2, y2 = face.bbox
    if min(x2 - x1, y2 - y1) < settings.FACE_MIN_PIXELS:
        raise FaceError(
            'The detected face is too small; use a closer or higher-resolution photo.',
            'face_too_small',
        )

    embedding = normalize(face.normed_embedding)
    info = {
        'faces_detected': len(faces),
        'detection_score': round(float(face.det_score), 4),
        # Reported in the coordinates of the submitted image.
        'face_box': [int(v) - pad for v in face.bbox],
        'padded_retry': pad > 0,
    }
    return embedding, info


def normalize(vector):
    """Return the L2-normalised float32 copy of a vector."""
    vector = np.asarray(vector, dtype=np.float32).ravel()
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise FaceError('Degenerate face embedding.', 'bad_embedding')
    return vector / norm


def cosine_similarity(a, b):
    """Cosine similarity of two embeddings, clamped to [-1, 1]."""
    return float(np.clip(np.dot(normalize(a), normalize(b)), -1.0, 1.0))


def to_confidence(cosine):
    """Map a raw cosine similarity onto a 0-100 'match confidence' percentage.

    ArcFace cosine scores are not percentages: genuine pairs typically land
    between ~0.4 and ~0.75, so comparing a raw cosine against 0.8 would reject
    nearly every true match. This applies a piecewise-linear calibration with
    three anchors:

        FACE_COS_FLOOR    -> 0%    (confidently different people)
        FACE_COS_THRESHOLD-> 50%   (the model's decision boundary)
        FACE_COS_CEILING  -> 100%  (unmistakably the same person)

    Tune the anchors in settings against your own photos; `calibrate_face`
    reports the cosine distribution for a labelled image set.
    """
    floor = settings.FACE_COS_FLOOR
    threshold = settings.FACE_COS_THRESHOLD
    ceiling = settings.FACE_COS_CEILING

    if cosine <= floor:
        percent = 0.0
    elif cosine < threshold:
        percent = 50.0 * (cosine - floor) / (threshold - floor)
    elif cosine < ceiling:
        percent = 50.0 + 50.0 * (cosine - threshold) / (ceiling - threshold)
    else:
        percent = 100.0
    return round(percent, 2)


def compare(embedding_a, embedding_b, min_confidence=None):
    """Compare two embeddings and decide whether they are the same person."""
    if min_confidence is None:
        min_confidence = settings.FACE_MIN_CONFIDENCE

    cosine = cosine_similarity(embedding_a, embedding_b)
    confidence = to_confidence(cosine)
    return {
        'cosine_similarity': round(cosine, 4),
        'confidence_percent': confidence,
        'required_confidence_percent': float(min_confidence),
        'above_model_threshold': cosine >= settings.FACE_COS_THRESHOLD,
        'is_match': confidence >= min_confidence,
    }


def pack_embedding(embedding):
    """Serialise an embedding for storage in a BinaryField."""
    return np.asarray(embedding, dtype=np.float32).tobytes()


def unpack_embedding(blob):
    """Deserialise an embedding stored by `pack_embedding`."""
    return np.frombuffer(bytes(blob), dtype=np.float32)
