import numpy as np
from mmcv.utils import assert_dict_has_keys
from numpy.testing import assert_array_equal

from mmaction.datasets.pipelines import ColorJitter


class TestColor:

    @staticmethod
    def test_color_jitter():
        imgs = list(
            np.random.randint(0, 255, size=(3, 112, 112, 3), dtype=np.uint8))
        results = dict(imgs=imgs)

        eig_val = np.array([55.46, 4.794, 1.148], dtype=np.float32)
        eig_vec = np.array([[-0.5675, 0.7192, 0.4009],
                            [-0.5808, -0.0045, -0.8140],
                            [-0.5836, -0.6948, 0.4203]],
                           dtype=np.float32)

        color_jitter = ColorJitter()
        assert_array_equal(color_jitter.eig_val, eig_val)
        assert_array_equal(color_jitter.eig_vec, eig_vec)
        assert color_jitter.alpha_std == 0.1
        assert color_jitter.color_space_aug is False
        color_jitter_results = color_jitter(results)
        target_keys = [
            'imgs', 'eig_val', 'eig_vec', 'alpha_std', 'color_space_aug'
        ]
        assert assert_dict_has_keys(color_jitter_results, target_keys)
        assert np.shape(color_jitter_results['imgs']) == (3, 112, 112, 3)
        assert_array_equal(color_jitter_results['eig_val'], eig_val)
        assert_array_equal(color_jitter_results['eig_vec'], eig_vec)
        assert color_jitter_results['alpha_std'] == 0.1
        assert color_jitter_results['color_space_aug'] is False

        custom_eig_val = np.ones(3, )
        custom_eig_vec = np.ones((3, 3))

        imgs = list(
            np.random.randint(0, 255, size=(3, 64, 80, 3), dtype=np.uint8))
        results = dict(imgs=imgs)
        custom_color_jitter = ColorJitter(True, 0.5, custom_eig_val,
                                          custom_eig_vec)
        assert_array_equal(color_jitter.eig_val, eig_val)
        assert_array_equal(color_jitter.eig_vec, eig_vec)
        assert custom_color_jitter.alpha_std == 0.5
        assert custom_color_jitter.color_space_aug is True
        custom_color_jitter_results = custom_color_jitter(results)
        assert np.shape(custom_color_jitter_results['imgs']) == (3, 64, 80, 3)
        assert_array_equal(custom_color_jitter_results['eig_val'],
                           custom_eig_val)
        assert_array_equal(custom_color_jitter_results['eig_vec'],
                           custom_eig_vec)
        assert custom_color_jitter_results['alpha_std'] == 0.5
        assert custom_color_jitter_results['color_space_aug'] is True

        color_jitter = ColorJitter()
        assert repr(color_jitter) == (f'{color_jitter.__class__.__name__}('
                                      f'color_space_aug={False}, '
                                      f'alpha_std={0.1}, '
                                      f'eig_val={eig_val}, '
                                      f'eig_vec={eig_vec})')
