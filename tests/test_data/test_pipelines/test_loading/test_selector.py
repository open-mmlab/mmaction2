import pytest
from tests.test_data.test_pipelines.test_loading.test_base_loading import \
    TestLoading

from mmaction.datasets.pipelines import FrameSelector


class TestSelector(TestLoading):

    def test_rawframe_selector(self):

        with pytest.warns(UserWarning):
            FrameSelector(io_backend='disk')
