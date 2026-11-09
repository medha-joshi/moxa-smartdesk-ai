from smartDeskAi.auth import authenticate, hash_password, verify_password


def test_hash_is_salted_and_verifies():
    first, second = hash_password("s3cret!"), hash_password("s3cret!")
    assert first != second  # random salt
    assert first.startswith("scrypt$")
    assert verify_password("s3cret!", first)
    assert verify_password("s3cret!", second)


def test_wrong_password_fails():
    assert not verify_password("wrong", hash_password("right"))


def test_malformed_or_foreign_hashes_fail_closed():
    assert not verify_password("x", "")
    assert not verify_password("x", "not-a-hash")
    assert not verify_password("x", "scrypt$zz$zz")  # not hex
    assert not verify_password("x", "bcrypt$00$00")  # unknown scheme


USERS = {
    "Asha.Rao@Synergy.com": {"name": "Asha Rao", "password_hash": hash_password("pw-asha")},
    "noname@synergy.com": {"password_hash": hash_password("pw-noname")},
}


def test_authenticate_success_normalizes_email():
    assert authenticate(USERS, "  ASHA.rao@synergy.com ", "pw-asha") == {
        "email": "asha.rao@synergy.com", "name": "Asha Rao",
    }


def test_authenticate_falls_back_to_email_as_name():
    assert authenticate(USERS, "noname@synergy.com", "pw-noname")["name"] == "noname@synergy.com"


def test_authenticate_rejects_wrong_password_and_unknown_email():
    assert authenticate(USERS, "asha.rao@synergy.com", "pw-noname") is None
    assert authenticate(USERS, "ghost@synergy.com", "pw-asha") is None
    assert authenticate({}, "ghost@synergy.com", "anything") is None
