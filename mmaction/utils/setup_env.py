# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmaction into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmaction default scope.
            If True, the global default scope will be set to `mmaction`, and all
            registries will build modules from mmaction's registry node. To
            understand more about the registry, please refer to
            https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import mmaction.core  # noqa: F401,F403
    import mmaction.datasets  # noqa: F401,F403
    import mmaction.metrics  # noqa: F401,F403
    import mmaction.models  # noqa: F401,F403

    if not init_default_scope:
        return

    current_scope = DefaultScope.get_current_instance()
    if current_scope is None:
        DefaultScope.get_instance('mmaction', scope_name='mmaction')
    elif current_scope.scope_name != 'mmaction':
        warnings.warn(f'The current default scope "{current_scope.scope_name}"'
                      ' is not "mmaction", `register_all_modules` will force the '
                      'current default scope to be "mmaction". If this is not '
                      'expected, please set `init_default_scope=False`.')
        # avoid name conflict
        new_instance_name = f'mmaction-{datetime.datetime.now()}'
        DefaultScope.get_instance(new_instance_name, scope_name='mmaction')
