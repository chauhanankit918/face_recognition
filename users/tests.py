from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.test import APITestCase

User = get_user_model()

VALID = {
    'username': 'alice',
    'email': 'alice@example.com',
    'password': 'sup3r-s3cret-pw',
    'password_confirm': 'sup3r-s3cret-pw',
}


class RegistrationTests(APITestCase):
    def test_register_creates_user_and_returns_token(self):
        response = self.client.post(reverse('register'), VALID)

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['user']['username'], 'alice')
        user = User.objects.get(username='alice')
        self.assertEqual(response.data['token'], Token.objects.get(user=user).key)
        self.assertTrue(user.check_password(VALID['password']))
        self.assertNotIn('password', response.data)

    def test_mismatched_passwords_are_rejected(self):
        response = self.client.post(
            reverse('register'), {**VALID, 'password_confirm': 'different-pw'}
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('password_confirm', response.data)
        self.assertFalse(User.objects.exists())

    def test_weak_password_is_rejected(self):
        response = self.client.post(
            reverse('register'),
            {**VALID, 'password': '12345', 'password_confirm': '12345'},
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('password', response.data)

    def test_duplicate_email_is_rejected(self):
        User.objects.create_user(username='bob', email=VALID['email'], password='x')

        response = self.client.post(reverse('register'), VALID)

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('email', response.data)


class TokenAuthTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='alice', email='alice@example.com', password='sup3r-s3cret-pw'
        )

    def test_login_returns_token(self):
        response = self.client.post(
            reverse('login'), {'username': 'alice', 'password': 'sup3r-s3cret-pw'}
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['token'], Token.objects.get(user=self.user).key)

    def test_login_with_bad_password_fails(self):
        response = self.client.post(
            reverse('login'), {'username': 'alice', 'password': 'wrong'}
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertFalse(Token.objects.exists())

    def test_inactive_user_cannot_log_in(self):
        User.objects.filter(pk=self.user.pk).update(is_active=False)

        response = self.client.post(
            reverse('login'), {'username': 'alice', 'password': 'sup3r-s3cret-pw'}
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_me_requires_a_token(self):
        self.assertEqual(
            self.client.get(reverse('me')).status_code, status.HTTP_401_UNAUTHORIZED
        )

    def test_me_returns_profile_for_token_holder(self):
        token = Token.objects.create(user=self.user)
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {token.key}')

        response = self.client.get(reverse('me'))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['email'], 'alice@example.com')

    def test_logout_revokes_the_token(self):
        token = Token.objects.create(user=self.user)
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {token.key}')

        self.assertEqual(
            self.client.post(reverse('logout')).status_code, status.HTTP_200_OK
        )
        self.assertFalse(Token.objects.filter(user=self.user).exists())
        self.assertEqual(
            self.client.get(reverse('me')).status_code, status.HTTP_401_UNAUTHORIZED
        )
