from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """Project user, keyed on a unique email address."""

    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=20, blank=True)

    REQUIRED_FIELDS = ['email']

    def __str__(self):
        return self.username
