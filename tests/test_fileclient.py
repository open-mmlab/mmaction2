import os.path as osp
import sys
from unittest.mock import MagicMock, patch

import mmcv
import pytest

from mmaction.utils import BaseStorageBackend, FileClient

sys.modules['ceph'] = MagicMock()
sys.modules['mc'] = MagicMock()


class MockS3Client(object):

    def Get(self, filepath):
        with open(filepath, 'rb') as f:
            content = f.read()
        return content


class MockMemcachedClient(object):

    def __init__(self, server_list_cfg, client_cfg):
        pass

    def Get(self, filepath, buffer):
        with open(filepath, 'rb') as f:
            buffer.content = f.read()


class TestFileClient(object):

    @classmethod
    def setup_class(cls):
        test_data_dir = osp.join(osp.dirname(__file__), 'data')
        cls.img_path = osp.join(test_data_dir, 'test.jpg')
        cls.text_path = osp.join(test_data_dir, 'frame_test_list.txt')

    def test_disk_backend(self):
        disk_backend = FileClient('disk')
        img_bytes = disk_backend.get(self.img_path)
        cur_frame = mmcv.imfrombytes(img_bytes)
        assert open(self.img_path, 'rb').read() == img_bytes
        assert cur_frame.shape == (240, 320, 3)

        value_buf = disk_backend.get_text(self.text_path)
        assert open(self.text_path, 'r').read() == value_buf

    @patch('ceph.S3Client', MockS3Client)
    def test_ceph_backend(self):
        ceph_backend = FileClient('ceph')

        with pytest.raises(NotImplementedError):
            # get_text is not implemented
            ceph_backend.get_text(self.text_path)

        img_bytes = ceph_backend.get(self.img_path)
        cur_frame = mmcv.imfrombytes(img_bytes)
        assert cur_frame.shape == (240, 320, 3)

    @patch('mc.MemcachedClient.GetInstance', MockMemcachedClient)
    @patch('mc.pyvector', MagicMock)
    @patch('mc.ConvertBuffer', lambda x: x.content)
    def test_memcached_backend(self):
        mc_cfg = dict(server_list_cfg='', client_cfg='', sys_path=None)
        mc_backend = FileClient('memcached', **mc_cfg)

        with pytest.raises(NotImplementedError):
            # get_text is not implemented
            mc_backend.get_text(self.text_path)

        img_bytes = mc_backend.get(self.img_path)
        cur_frame = mmcv.imfrombytes(img_bytes)
        assert cur_frame.shape == (240, 320, 3)

    def test_register_backend(self):
        with pytest.raises(TypeError):
            # unsupported backend
            class TestClass1(object):
                pass

            FileClient.register_backend('TestClass1', TestClass1)

        with pytest.raises(TypeError):
            # unsupported backend
            FileClient.register_backend('int', 0)

        class ExampleBackend(BaseStorageBackend):

            def get(self, filepath):
                return filepath

            def get_text(self, filepath):
                return filepath

        FileClient.register_backend('example', ExampleBackend)
        example_backend = FileClient('example')
        assert example_backend.get(self.img_path) == self.img_path
        assert example_backend.get_text(self.text_path) == self.text_path
        assert 'example' in FileClient._backends

    def test_error(self):
        with pytest.raises(ValueError):
            FileClient('hadoop')
