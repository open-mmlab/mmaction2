from mmaction.datasets.pipelines.augmentations import (combine_quadruple,
                                                       flip_quadruple)


class TestQuadrupleOps:

    def test_combine_quadruple(self):
        a = (0.1, 0.1, 0.5, 0.5)
        b = (0.3, 0.3, 0.7, 0.7)
        res = combine_quadruple(a, b)
        assert res == (0.25, 0.25, 0.35, 0.35)

    def test_flip_quadruple(self):
        a = (0.1, 0.1, 0.5, 0.5)
        res = flip_quadruple(a)
        assert res == (0.4, 0.1, 0.5, 0.5)
