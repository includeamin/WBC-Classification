import wbc_classification


def test_version_is_a_string():
    assert isinstance(wbc_classification.__version__, str)
    assert wbc_classification.__version__.count(".") == 2
