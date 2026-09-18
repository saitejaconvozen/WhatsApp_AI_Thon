from templatelab.policy import assess


def review(body, **extra):
    return assess({"body": body, "purpose": "billing", "relationship_confirmed": True, **extra})


def test_supported_invoice_is_only_a_candidate():
    result = review("Your invoice {{id}} is due on {{date}}.")
    assert result["category"] == "UTILITY_CANDIDATE"
    assert not result["rewrite"]["available"]


def test_rewrite_requires_context():
    result = review("Your invoice {{id}} is ready. Upgrade today.", relationship_confirmed=False)
    assert result["category"] == "LIKELY_MARKETING"
    assert not result["rewrite"]["available"]


def test_limited_rewrite_preserves_transaction_and_placeholders():
    body = "Your invoice {{id}} for {{amount}} is due on {{date}}. Upgrade today."
    result = review(body)
    assert result["rewrite"]["available"]
    assert result["rewrite"]["components"]["body"] == "Your invoice {{id}} for {{amount}} is due on {{date}}."
    assert result["rewrite"]["removed"] == [{"component": "body", "text": "Upgrade today."}]


def test_mixed_sentence_is_not_silently_deleted():
    result = review("Your invoice {{id}} is ready with an upgrade offer.")
    assert not result["rewrite"]["available"]


def test_pure_promotion_does_not_get_a_fabricated_transaction():
    result = review("Get 20% off. Shop now.")
    assert result["category"] == "LIKELY_MARKETING"
    assert not result["rewrite"]["available"]


def test_promotion_in_button_is_detected():
    result = review("Your invoice {{id}} is ready.", buttons="View invoice\nShop now")
    assert result["category"] == "LIKELY_MARKETING"
    assert result["rewrite"]["components"]["buttons"] == "View invoice"


def test_authentication_is_separate():
    assert review("Your verification code is {{code}}.")["category"] == "AUTHENTICATION"


def test_unsupported_format_needs_review():
    assert review("Your invoice {{id}} is ready.", format="MEDIA")["category"] == "NEEDS_REVIEW"
