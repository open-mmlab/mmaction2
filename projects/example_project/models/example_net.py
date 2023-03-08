from mmaction.models import ResNet
from mmaction.registry import MODELS


# Register your model to the `MODELS`.
@MODELS.register_module()
class ExampleNet(ResNet):
    """Implements an example backbone.

    Implement the backbone network just like a normal pytorch network.
    """

    def __init__(self, **kwargs) -> None:
        print('#############################\n'
              '#      Hello MMAction2!     #\n'
              '#############################')
        super().__init__(**kwargs)

    def forward(self, x):
        """Defines the computation performed at every call."""
        return super().forward(x)
