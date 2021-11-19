from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class TSMHeadRandomInit(Hook):
    def __init__(self):
        super().__init__() 
    
    def before_run(self, runner):
        runner.model.module.cls_head.init_weights() 