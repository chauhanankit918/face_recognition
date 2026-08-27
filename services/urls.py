from django.urls import path

from .views import (
    EnrollFaceView,
    FaceProfileView,
    VerificationHistoryView,
    VerifyFaceView,
)

urlpatterns = [
    path('enroll/', EnrollFaceView.as_view(), name='face-enroll'),
    path('verify/', VerifyFaceView.as_view(), name='face-verify'),
    path('profile/', FaceProfileView.as_view(), name='face-profile'),
    path('attempts/', VerificationHistoryView.as_view(), name='face-attempts'),
]
