import io
import shutil
import tempfile
from unittest import mock

import numpy as np
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, override_settings
from django.urls import reverse
from PIL import Image
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.test import APITestCase

from services import face_engine
from services.models import FaceProfile, VerificationAttempt

User = get_user_model()


def png_upload(name='face.png', size=(120, 120)):
    """A real (faceless) PNG, enough to satisfy ImageField validation."""
    buffer = io.BytesIO()
    Image.new('RGB', size, (128, 130, 132)).save(buffer, format='PNG')
    return SimpleUploadedFile(name, buffer.getvalue(), content_type='image/png')


def fake_embedding(seed):
    rng = np.random.default_rng(seed)
    return face_engine.normalize(rng.standard_normal(512))


class ConfidenceCalibrationTests(SimpleTestCase):
    """The cosine -> percentage mapping, independent of any model."""

    def test_anchors_map_to_expected_percentages(self):
        self.assertEqual(face_engine.to_confidence(face_engine.settings.FACE_COS_FLOOR), 0.0)
        self.assertEqual(face_engine.to_confidence(face_engine.settings.FACE_COS_THRESHOLD), 50.0)
        self.assertEqual(face_engine.to_confidence(face_engine.settings.FACE_COS_CEILING), 100.0)

    def test_mapping_is_monotonic_and_bounded(self):
        values = [face_engine.to_confidence(c) for c in np.linspace(-1.0, 1.0, 201)]
        self.assertEqual(values, sorted(values))
        self.assertEqual(min(values), 0.0)
        self.assertEqual(max(values), 100.0)

    def test_identical_embeddings_are_a_perfect_match(self):
        embedding = fake_embedding(1)
        result = face_engine.compare(embedding, embedding)

        self.assertAlmostEqual(result['cosine_similarity'], 1.0, places=3)
        self.assertEqual(result['confidence_percent'], 100.0)
        self.assertTrue(result['is_match'])

    def test_unrelated_embeddings_do_not_match(self):
        result = face_engine.compare(fake_embedding(1), fake_embedding(2))

        self.assertFalse(result['is_match'])
        self.assertLess(result['confidence_percent'], 80.0)

    @override_settings(FACE_MIN_CONFIDENCE=80.0)
    def test_min_confidence_override_is_honoured(self):
        a, b = fake_embedding(1), fake_embedding(2)
        self.assertFalse(face_engine.compare(a, b)['is_match'])
        self.assertTrue(face_engine.compare(a, b, min_confidence=0)['is_match'])

    def test_pack_round_trips_an_embedding(self):
        embedding = fake_embedding(7)
        restored = face_engine.unpack_embedding(face_engine.pack_embedding(embedding))

        np.testing.assert_allclose(embedding, restored, rtol=1e-6)

    def test_undecodable_bytes_raise_face_error(self):
        with self.assertRaises(face_engine.FaceError) as ctx:
            face_engine.decode_image(b'not an image')
        self.assertEqual(ctx.exception.code, 'invalid_image')


class FaceAPITestCase(APITestCase):
    """Shared auth setup; the model itself is patched out for speed."""

    @classmethod
    def setUpClass(cls):
        cls._media = tempfile.mkdtemp()
        cls._override = override_settings(MEDIA_ROOT=cls._media)
        cls._override.enable()
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls._override.disable()
        shutil.rmtree(cls._media, ignore_errors=True)

    def setUp(self):
        self.user = User.objects.create_user(
            username='alice', email='alice@example.com', password='pw'
        )
        self.token = Token.objects.create(user=self.user)
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {self.token.key}')

    def patch_engine(self, embedding, faces_detected=1):
        info = {
            'faces_detected': faces_detected,
            'detection_score': 0.99,
            'face_box': [1, 2, 3, 4],
            'padded_retry': False,
        }
        return mock.patch.object(
            face_engine, 'get_embedding', return_value=(embedding, info)
        )

    def enroll(self, embedding):
        with self.patch_engine(embedding):
            return self.client.post(
                reverse('face-enroll'), {'image': png_upload()}, format='multipart'
            )


class EnrollTests(FaceAPITestCase):
    def test_enrolment_requires_authentication(self):
        self.client.credentials()

        response = self.client.post(
            reverse('face-enroll'), {'image': png_upload()}, format='multipart'
        )

        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_enrolment_stores_the_embedding(self):
        embedding = fake_embedding(1)

        response = self.enroll(embedding)

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(response.data['created'])
        profile = FaceProfile.objects.get(user=self.user)
        np.testing.assert_allclose(profile.get_embedding(), embedding, rtol=1e-6)
        self.assertEqual(profile.detection_score, 0.99)

    def test_re_enrolment_replaces_the_existing_profile(self):
        self.enroll(fake_embedding(1))

        response = self.enroll(fake_embedding(2))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(response.data['created'])
        self.assertEqual(FaceProfile.objects.filter(user=self.user).count(), 1)
        np.testing.assert_allclose(
            FaceProfile.objects.get(user=self.user).get_embedding(),
            fake_embedding(2), rtol=1e-6,
        )

    def test_faceless_image_is_rejected(self):
        with mock.patch.object(
            face_engine, 'get_embedding',
            side_effect=face_engine.FaceError('No face detected.', 'no_face'),
        ):
            response = self.client.post(
                reverse('face-enroll'), {'image': png_upload()}, format='multipart'
            )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data['code'], 'no_face')
        self.assertFalse(FaceProfile.objects.exists())

    def test_non_image_upload_is_rejected(self):
        upload = SimpleUploadedFile('x.png', b'not an image', content_type='image/png')

        response = self.client.post(
            reverse('face-enroll'), {'image': upload}, format='multipart'
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('image', response.data)


class VerifyTests(FaceAPITestCase):
    def test_verification_requires_authentication(self):
        self.client.credentials()

        response = self.client.post(
            reverse('face-verify'), {'image': png_upload()}, format='multipart'
        )

        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_verification_without_enrolment_conflicts(self):
        with self.patch_engine(fake_embedding(1)):
            response = self.client.post(
                reverse('face-verify'), {'image': png_upload()}, format='multipart'
            )

        self.assertEqual(response.status_code, status.HTTP_409_CONFLICT)
        self.assertEqual(response.data['code'], 'not_enrolled')

    def test_same_face_matches_and_performs_the_action(self):
        embedding = fake_embedding(1)
        self.enroll(embedding)

        with self.patch_engine(embedding):
            response = self.client.post(
                reverse('face-verify'), {'image': png_upload()}, format='multipart'
            )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['is_match'])
        self.assertEqual(response.data['confidence_percent'], 100.0)
        self.assertEqual(response.data['required_confidence_percent'], 80.0)
        self.assertTrue(response.data['action']['performed'])

    def test_different_face_does_not_match_and_skips_the_action(self):
        self.enroll(fake_embedding(1))

        with self.patch_engine(fake_embedding(2)):
            response = self.client.post(
                reverse('face-verify'), {'image': png_upload()}, format='multipart'
            )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(response.data['is_match'])
        self.assertLess(response.data['confidence_percent'], 80.0)
        self.assertFalse(response.data['action']['performed'])

    def test_action_is_not_called_when_confidence_is_below_the_gate(self):
        self.enroll(fake_embedding(1))

        with self.patch_engine(fake_embedding(2)), mock.patch(
            'services.views.perform_verified_action'
        ) as action:
            self.client.post(
                reverse('face-verify'), {'image': png_upload()}, format='multipart'
            )

        action.assert_not_called()

    def test_min_confidence_override_can_tighten_the_gate(self):
        embedding = fake_embedding(1)
        self.enroll(embedding)

        with self.patch_engine(embedding):
            response = self.client.post(
                reverse('face-verify'),
                {'image': png_upload(), 'min_confidence': 100},
                format='multipart',
            )

        self.assertEqual(response.data['required_confidence_percent'], 100.0)
        self.assertTrue(response.data['is_match'])

    def test_out_of_range_min_confidence_is_rejected(self):
        self.enroll(fake_embedding(1))

        response = self.client.post(
            reverse('face-verify'),
            {'image': png_upload(), 'min_confidence': 150},
            format='multipart',
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_users_cannot_verify_against_another_users_face(self):
        """Bob's own profile is used even when Alice is enrolled."""
        alice_face = fake_embedding(1)
        self.enroll(alice_face)

        bob = User.objects.create_user(username='bob', email='b@example.com', password='pw')
        bob_token = Token.objects.create(user=bob)
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {bob_token.key}')

        with self.patch_engine(alice_face):
            response = self.client.post(
                reverse('face-verify'), {'image': png_upload()}, format='multipart'
            )

        self.assertEqual(response.status_code, status.HTTP_409_CONFLICT)
        self.assertEqual(response.data['code'], 'not_enrolled')


class AuditTrailTests(FaceAPITestCase):
    def test_successful_and_failed_attempts_are_recorded(self):
        embedding = fake_embedding(1)
        self.enroll(embedding)

        with self.patch_engine(embedding):
            self.client.post(reverse('face-verify'), {'image': png_upload()}, format='multipart')
        with self.patch_engine(fake_embedding(2)):
            self.client.post(reverse('face-verify'), {'image': png_upload()}, format='multipart')

        attempts = VerificationAttempt.objects.filter(user=self.user)
        self.assertEqual(attempts.count(), 2)
        self.assertEqual(attempts.filter(is_match=True).count(), 1)

    def test_detection_failures_are_recorded_with_an_error_code(self):
        self.enroll(fake_embedding(1))

        with mock.patch.object(
            face_engine, 'get_embedding',
            side_effect=face_engine.FaceError('2 faces.', 'multiple_faces'),
        ):
            response = self.client.post(
                reverse('face-verify'), {'image': png_upload()}, format='multipart'
            )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        attempt = VerificationAttempt.objects.get(user=self.user)
        self.assertEqual(attempt.error_code, 'multiple_faces')
        self.assertFalse(attempt.is_match)

    def test_history_only_exposes_the_callers_attempts(self):
        VerificationAttempt.objects.create(user=self.user, is_match=True, confidence_percent=95)
        other = User.objects.create_user(username='bob', email='b@example.com', password='pw')
        VerificationAttempt.objects.create(user=other, is_match=True, confidence_percent=99)

        response = self.client.get(reverse('face-attempts'))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['count'], 1)
        self.assertEqual(response.data['results'][0]['confidence_percent'], 95.0)


class ProfileTests(FaceAPITestCase):
    def test_profile_is_404_before_enrolment(self):
        response = self.client.get(reverse('face-profile'))

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(response.data['code'], 'not_enrolled')

    def test_profile_is_readable_and_deletable(self):
        self.enroll(fake_embedding(1))

        self.assertEqual(self.client.get(reverse('face-profile')).status_code, status.HTTP_200_OK)
        self.assertEqual(
            self.client.delete(reverse('face-profile')).status_code,
            status.HTTP_204_NO_CONTENT,
        )
        self.assertFalse(FaceProfile.objects.exists())
        self.assertEqual(
            self.client.delete(reverse('face-profile')).status_code,
            status.HTTP_404_NOT_FOUND,
        )
