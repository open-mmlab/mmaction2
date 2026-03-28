"""
Description:
Version: 1.0
Author: ZhuYichen
Date: 2023-09-10 17:03:05
LastEditors: ZhuYichen
LastEditTime: 2023-09-11 17:57:32
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


class Qwen:

    def __init__(self, model_path):
        self.model_path = model_path

    def init_model(self):
        # Note: The default behavior now has injection attack prevention off.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, device_map='cuda', trust_remote_code=True).eval()

        # Specify hyperparameters for generation
        self.model.generation_config = GenerationConfig.from_pretrained(
            self.model_path, trust_remote_code=True)

    def inference(self, image, question):
        # 1st dialogue turn
        query = self.tokenizer.from_list_format([
            {
                'image': image
            },  # Either a local path or an url
            {
                'text':
                '''该图片是从一个视频片段中截取的一帧，问题可能与这一帧图片无关。
                因此如果图片中不包含问题中提到的信息，直接回答“缺失信息”。''' + f'\n问题：{question}'
            },
        ])
        response, history = self.model.chat(
            self.tokenizer, query=query, history=None)
        print(response)
        return response
