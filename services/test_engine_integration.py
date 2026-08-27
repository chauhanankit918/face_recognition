"""End-to-end checks against the real InsightFace model.

These load the ~330 MB `buffalo_l` pack, so they are slower than the mocked
API tests. Run them on their own with:

    manage.py test services.test_engine_integration
"""
import itertools
from pathlib import Path

import cv2
import insightface
import numpy as np
from django.test import SimpleTestCase, override_settings

from services import face_engine

SAMPLES = Path(insightface.__file__).resolve().parent / 'data' / 'images'
GROUP_PHOTO = SAMPLES / 't1.jpg'
CROPPED_PORTRAIT = SAMPLES / 'Tom_Hanks_54745.png'


def encode(image, quality=95):
    return cv2.imencode(
        '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )[1].tobytes()


def astronaut():
    from skimage import data
    return cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)


class DetectionTests(SimpleTestCase):
    def test_face_is_detected_in_a_normal_photo(self):
        _, info = face_engine.get_embedding(encode(astronaut()))

        self.assertEqual(info['faces_detected'], 1)
        self.assertGreater(info['detection_score'], 0.5)
        self.assertFalse(info['padded_retry'])

    def test_tightly_cropped_portrait_falls_back_to_a_padded_retry(self):
        _, info = face_engine.get_embedding(CROPPED_PORTRAIT.read_bytes())

        self.assertEqual(info['faces_detected'], 1)
        self.assertTrue(info['padded_retry'])

    def test_image_without_a_face_is_rejected(self):
        from skimage import data
        cat = cv2.cvtColor(data.chelsea(), cv2.COLOR_RGB2BGR)

        with self.assertRaises(face_engine.FaceError) as ctx:
            face_engine.get_embedding(encode(cat))

        self.assertEqual(ctx.exception.code, 'no_face')

    def test_group_photo_is_rejected_when_multiple_faces_are_disallowed(self):
        with self.assertRaises(face_engine.FaceError) as ctx:
            face_engine.get_embedding(GROUP_PHOTO.read_bytes())

        self.assertEqual(ctx.exception.code, 'multiple_faces')

    @override_settings(FACE_REJECT_MULTIPLE=False)
    def test_group_photo_uses_the_largest_face_when_allowed(self):
        _, info = face_engine.get_embedding(GROUP_PHOTO.read_bytes())

        self.assertGreater(info['faces_detected'], 1)

    def test_tiny_face_is_rejected(self):
        small = cv2.resize(astronaut(), None, fx=0.4, fy=0.4)

        with self.assertRaises(face_engine.FaceError) as ctx:
            face_engine.get_embedding(encode(small))

        self.assertEqual(ctx.exception.code, 'face_too_small')

    def test_embedding_is_512_dimensional_and_unit_norm(self):
        embedding, _ = face_engine.get_embedding(encode(astronaut()))

        self.assertEqual(embedding.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=4)


class SamePersonTests(SimpleTestCase):
    """A degraded photo of a person must still verify against the original."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.original = cv2.imread(str(CROPPED_PORTRAIT))
        cls.reference, _ = face_engine.get_embedding(CROPPED_PORTRAIT.read_bytes())

    def assert_matches(self, image, label):
        embedding, _ = face_engine.get_embedding(encode(image))
        result = face_engine.compare(embedding, self.reference)

        self.assertTrue(
            result['is_match'],
            f'{label}: only {result["confidence_percent"]}% confident '
            f'(cosine {result["cosine_similarity"]})',
        )

    def test_heavy_jpeg_compression_still_matches(self):
        degraded = cv2.imdecode(
            np.frombuffer(encode(self.original, quality=25), np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assert_matches(degraded, 'jpeg q25')

    def test_blur_still_matches(self):
        self.assert_matches(cv2.GaussianBlur(self.original, (7, 7), 3), 'blur')

    def test_underexposure_still_matches(self):
        self.assert_matches(
            cv2.convertScaleAbs(self.original, alpha=0.55, beta=-15), 'dark'
        )

    def test_grayscale_still_matches(self):
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        self.assert_matches(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 'grayscale')

    def test_slight_skew_still_matches(self):
        height, width = self.original.shape[:2]
        skew = np.float32([[1, 0.15, 0], [0, 1, 0]])
        self.assert_matches(
            cv2.warpAffine(self.original, skew, (width, height)), 'skew'
        )


class DifferentPeopleTests(SimpleTestCase):
    """Distinct people must never clear the confidence gate."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        group = cv2.imread(str(GROUP_PHOTO))
        faces, _ = face_engine.detect_faces(group)
        cls.embeddings = [face_engine.normalize(f.normed_embedding) for f in faces]
        cls.embeddings.append(
            face_engine.get_embedding(CROPPED_PORTRAIT.read_bytes())[0]
        )
        cls.embeddings.append(face_engine.get_embedding(encode(astronaut()))[0])

    def test_the_fixture_really_contains_several_people(self):
        self.assertGreaterEqual(len(self.embeddings), 5)

    def test_no_pair_of_different_people_is_accepted(self):
        accepted = [
            face_engine.compare(a, b)
            for a, b in itertools.combinations(self.embeddings, 2)
            if face_engine.compare(a, b)['is_match']
        ]

        self.assertEqual(accepted, [], f'false accepts: {accepted}')

    def test_different_people_stay_below_the_model_threshold(self):
        cosines = [
            face_engine.cosine_similarity(a, b)
            for a, b in itertools.combinations(self.embeddings, 2)
        ]

        self.assertLess(
            max(cosines), face_engine.settings.FACE_COS_THRESHOLD,
            'an impostor pair reached the decision threshold',
        )
