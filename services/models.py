from django.conf import settings
from django.db import models

from . import face_engine


class FaceProfile(models.Model):
    """The reference face on file for a user.

    The 512-d embedding is computed once at enrolment and cached here, so a
    verification request only has to run the model over the incoming image.
    """

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='face_profile',
    )
    image = models.ImageField(upload_to='face_profiles/')
    embedding = models.BinaryField(editable=False)
    model_name = models.CharField(max_length=50, default=settings.FACE_MODEL_NAME)
    detection_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'FaceProfile<{self.user}>'

    def get_embedding(self):
        return face_engine.unpack_embedding(self.embedding)

    def set_embedding(self, embedding):
        self.embedding = face_engine.pack_embedding(embedding)


class VerificationAttempt(models.Model):
    """Audit trail of every verification call."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='verification_attempts',
    )
    cosine_similarity = models.FloatField(null=True, blank=True)
    confidence_percent = models.FloatField(null=True, blank=True)
    required_confidence_percent = models.FloatField(null=True, blank=True)
    is_match = models.BooleanField(default=False)
    error_code = models.CharField(max_length=40, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        outcome = 'match' if self.is_match else (self.error_code or 'no-match')
        return f'{self.user} @ {self.created_at:%Y-%m-%d %H:%M:%S} ({outcome})'
