import mmaction


def test_version():
    version = mmaction.__version__
    assert isinstance(version, str)
    assert isinstance(mmaction.short_version, str)
    assert mmaction.short_version in version and '+' in version
