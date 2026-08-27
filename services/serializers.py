from rest_framework import serializers

from .models import FaceProfile, VerificationAttempt


class FaceImageSerializer(serializers.Serializer):
    """An uploaded face image, size-checked before it reaches the model."""

    image = serializers.ImageField()

    MAX_BYTES = 10 * 1024 * 1024

    def validate_image(self, value):
        if value.size > self.MAX_BYTES:
            raise serializers.ValidationError(
                f'Image must be at most {self.MAX_BYTES // (1024 * 1024)} MB.'
            )
        return value


class VerifySerializer(FaceImageSerializer):
    """Verification input; `min_confidence` optionally overrides the default gate."""

    min_confidence = serializers.FloatField(
        required=False, min_value=0, max_value=100
    )


class FaceProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = FaceProfile
        fields = ['image', 'model_name', 'detection_score', 'created_at', 'updated_at']
        read_only_fields = fields


class VerificationAttemptSerializer(serializers.ModelSerializer):
    class Meta:
        model = VerificationAttempt
        fields = ['id', 'cosine_similarity', 'confidence_percent',
                  'required_confidence_percent', 'is_match', 'error_code',
                  'created_at']
        read_only_fields = fields
