"""Business logic that runs once a face has been verified.

Replace the body of `perform_verified_action` with whatever the successful
verification should actually trigger (mark attendance, release a document,
approve a KYC record, unlock a transaction, ...).
"""
import logging

logger = logging.getLogger(__name__)


def perform_verified_action(user, result):
    """Run the post-verification workflow for `user`.

    Args:
        user: the authenticated user whose face matched.
        result: the comparison dict from `face_engine.compare`.

    Returns:
        A JSON-serialisable dict describing what was done. It is echoed back to
        the client under the `action` key.
    """
    logger.info(
        'Face verified for %s at %.2f%% confidence',
        user, result['confidence_percent'],
    )

    # TODO: replace with the real workflow.
    return {
        'performed': True,
        'name': 'face_verified',
        'detail': f'Verified action executed for {user.username}.',
    }
