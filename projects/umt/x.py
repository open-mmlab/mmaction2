from mmengine import Config
from mmaction.registry import MODELS
from mmaction.utils import register_all_modules
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
register_all_modules()
x = Config.fromfile('./configs/k400.py')


model = MODELS.build(x.model)
