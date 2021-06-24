from mmaction.datasets.pipelines.augmentations import (_combine_quadruple,
                                                       _flip_quadruple)


class TestQuadrupleOps:

    @staticmethod
    def test_combine_quadruple():
        a = (0.1, 0.1, 0.5, 0.5)
        b = (0.3, 0.3, 0.7, 0.7)
        res = _combine_quadruple(a, b)
        assert res == (0.25, 0.25, 0.35, 0.35)

    @staticmethod
    def test_flip_quadruple():
        a = (0.1, 0.1, 0.5, 0.5)
        res = _flip_quadruple(a)
        assert res == (0.4, 0.1, 0.5, 0.5)
