from .builder import DATASETS
from .video_dataset import VideoDataset


@DATASETS.register_module()
class ImageDataset(VideoDataset):
    """Image dataset for action recognition, used in the Project OmniSource.

    The dataset loads image list and apply specified transforms to return a
    dict containing the image tensors and other information. For the
    ImageDataset

    The ann_file is a text file with multiple lines, and each line indicates
    the image path and the image label, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        path/to/image1.jpg 1
        path/to/image2.jpg 1
        path/to/image3.jpg 2
        path/to/image4.jpg 2
        path/to/image5.jpg 3
        path/to/image6.jpg 3

    Example of a multi-class annotation file:

    .. code-block:: txt

        path/to/image1.jpg 1 3 5
        path/to/image2.jpg 1 2
        path/to/image3.jpg 2
        path/to/image4.jpg 2 4 6 8
        path/to/image5.jpg 3
        path/to/image6.jpg 3

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self, ann_file, pipeline, **kwargs):
        super().__init__(ann_file, pipeline, start_index=None, **kwargs)
        # use `start_index=None` to indicate it is for `ImageDataset`
