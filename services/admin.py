from django.contrib import admin

from .models import FaceProfile, VerificationAttempt


@admin.register(FaceProfile)
class FaceProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'model_name', 'detection_score', 'updated_at']
    search_fields = ['user__username', 'user__email']
    readonly_fields = ['embedding', 'created_at', 'updated_at']


@admin.register(VerificationAttempt)
class VerificationAttemptAdmin(admin.ModelAdmin):
    list_display = ['user', 'confidence_percent', 'cosine_similarity', 'is_match',
                    'error_code', 'created_at']

    list_filter = ['is_match', 'error_code']
    search_fields = ['user__username', 'user__email']
    readonly_fields = [f.name for f in VerificationAttempt._meta.fields]
