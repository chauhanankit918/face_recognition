from django.conf import settings
from rest_framework import status
from rest_framework.generics import ListAPIView
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from . import face_engine
from .actions import perform_verified_action
from .models import FaceProfile, VerificationAttempt
from .serializers import (
    FaceImageSerializer,
    FaceProfileSerializer,
    VerificationAttemptSerializer,
    VerifySerializer,
)


class EnrollFaceView(APIView):
    """POST /api/face/enroll/ - store (or replace) the caller's reference face."""

    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        serializer = FaceImageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        image = serializer.validated_data['image']

        try:
            embedding, info = face_engine.get_embedding(image.read())
        except face_engine.FaceError as exc:
            return Response(
                {'detail': exc.message, 'code': exc.code},
                status=status.HTTP_400_BAD_REQUEST,
            )

        image.seek(0)
        profile, created = FaceProfile.objects.get_or_create(
            user=request.user,
            defaults={'image': image, 'embedding': b''},
        )
        if not created:
            profile.image = image
        profile.set_embedding(embedding)
        profile.model_name = settings.FACE_MODEL_NAME
        profile.detection_score = info['detection_score']
        profile.save()

        return Response(
            {'detail': 'Face enrolled.', 'created': created,
             'profile': FaceProfileSerializer(profile).data, **info},
            status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
        )


class VerifyFaceView(APIView):
    """POST /api/face/verify/ - compare an uploaded face against the stored one.

    On a match at or above the confidence gate (80% by default), the
    post-verification workflow in `services.actions` is executed.
    """

    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    throttle_scope = 'face_verify'

    def post(self, request):
        serializer = VerifySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        image = serializer.validated_data['image']
        min_confidence = serializer.validated_data.get('min_confidence')

        try:
            profile = request.user.face_profile
        except FaceProfile.DoesNotExist:
            return Response(
                {'detail': 'No reference face on file. Enrol one first.',
                 'code': 'not_enrolled'},
                status=status.HTTP_409_CONFLICT,
            )

        try:
            embedding, info = face_engine.get_embedding(image.read())
        except face_engine.FaceError as exc:
            VerificationAttempt.objects.create(
                user=request.user, error_code=exc.code, is_match=False
            )
            return Response(
                {'detail': exc.message, 'code': exc.code, 'is_match': False},
                status=status.HTTP_400_BAD_REQUEST,
            )

        result = face_engine.compare(
            embedding, profile.get_embedding(), min_confidence=min_confidence
        )
        VerificationAttempt.objects.create(
            user=request.user,
            cosine_similarity=result['cosine_similarity'],
            confidence_percent=result['confidence_percent'],
            required_confidence_percent=result['required_confidence_percent'],
            is_match=result['is_match'],
        )

        payload = {**result, **info}
        if result['is_match']:
            payload['action'] = perform_verified_action(request.user, result)
            payload['detail'] = 'Face verified; action performed.'
        else:
            payload['action'] = {'performed': False}
            payload['detail'] = (
                f"Face did not match to the required "
                f"{result['required_confidence_percent']:.0f}% confidence."
            )

        return Response(payload)


class FaceProfileView(APIView):
    """GET /api/face/profile/ - inspect the stored reference face.
    DELETE /api/face/profile/ - remove it."""

    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            profile = request.user.face_profile
        except FaceProfile.DoesNotExist:
            return Response(
                {'detail': 'No reference face on file.', 'code': 'not_enrolled'},
                status=status.HTTP_404_NOT_FOUND,
            )
        return Response(FaceProfileSerializer(profile).data)

    def delete(self, request):
        deleted, _ = FaceProfile.objects.filter(user=request.user).delete()
        if not deleted:
            return Response(
                {'detail': 'No reference face on file.', 'code': 'not_enrolled'},
                status=status.HTTP_404_NOT_FOUND,
            )
        return Response(status=status.HTTP_204_NO_CONTENT)


class VerificationHistoryView(ListAPIView):
    """GET /api/face/attempts/ - the caller's verification history."""

    permission_classes = [IsAuthenticated]
    serializer_class = VerificationAttemptSerializer

    def get_queryset(self):
        return self.request.user.verification_attempts.all()
