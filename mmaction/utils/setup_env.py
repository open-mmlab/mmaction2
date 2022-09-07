# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmaction into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmaction default
            scope. If True, the global default scope will be set to `mmaction`,
            and all registries will build modules from mmaction's registry
            node. To understand more about the registry, please refer to
            https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """
    import mmaction.datasets  # noqa: F401,F403
    import mmaction.engine  # noqa: F401,F403
    import mmaction.evaluation  # noqa: F401,F403
    import mmaction.models  # noqa: F401,F403
    import mmaction.structures  # noqa: F401,F403
    import mmaction.visualization  # noqa: F401,F403

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('mmaction')
        if never_created:
            DefaultScope.get_instance('mmaction', scope_name='mmaction')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmaction':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmaction", '
                          '`register_all_modules` will force set the current'
                          'default scope to "mmaction". If this is not as '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'mmaction-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='mmaction')
