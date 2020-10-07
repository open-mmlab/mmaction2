from mmaction.version import parse_version_info


def test_parse_version_info():
    assert parse_version_info('0.2.16') == (0, 2, 16)
    assert parse_version_info('1.2.3') == (1, 2, 3)
    assert parse_version_info('1.2.3rc0') == (1, 2, 3, 'rc0')
    assert parse_version_info('1.2.3rc1') == (1, 2, 3, 'rc1')
    assert parse_version_info('1.0rc0') == (1, 0, 'rc0')
