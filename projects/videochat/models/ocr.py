import ast
import math
import os
import subprocess
from collections import defaultdict

import openai
from paddleocr import PaddleOCR

from projects.videochat.util import loadvideo_decord_origin

openai.api_key = os.getenv('OPENAI_API_KEY')


def compute_square(pos):
    return (pos[2] - pos[0]) * (pos[3] - pos[1])


def remove_duplicates_by_text_key(lst, value):
    seen_text_values = set()
    result = []

    for obj in lst:
        text_value = obj.get(value)

        if text_value not in seen_text_values:
            seen_text_values.add(text_value)
            result.append(obj)

    return result


class ProcessOCR:

    def __init__(self):
        self.video_path = None

    def inference(self, video_path, data=None):
        self.video_path = video_path
        ocr = PaddleOCR(
            use_angle_cls=True, lang='ch'
        )  # need to run only once to download and load model into memory
        if data is None:
            try:
                data = loadvideo_decord_origin(video_path)
            except Exception as e:
                print(e)
                return ''
        text_result = list()

        for i in range(data.shape[0]):
            result = ocr.ocr(data[i], cls=True)
            text_per_image = list()
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    if line[1][1] < 0.8:
                        continue
                    text = dict()
                    text['pos'] = [
                        line[0][0][0], line[0][0][1], line[0][2][0],
                        line[0][2][1]
                    ]
                    text['text'] = line[1][0]
                    text_per_image.append(text)
            if len(text_per_image) > 0:
                text_result.append({'begin': i, 'text': text_per_image})
        return text_result

    def merge(self, features):
        # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
        # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
        # video_path = self.video_path
        video_width, video_height = self.get_video_resolution_ffmpeg()
        # dense_with_pos = ''
        # ocr_subtitle_result = ''
        # ocr_subtitle_result2 = ''
        dense_with_ocr, ocr_subtitle = self.find_text_in_dense(
            features, video_width, video_height)
        return dense_with_ocr, ocr_subtitle

    def get_video_resolution_ffmpeg(self):
        video_path = self.video_path
        try:
            # 调用FFmpeg命令行获取视频宽度和高度信息
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height', '-of', 'csv=p=0',
                video_path
            ]
            result = subprocess.check_output(
                cmd, stderr=subprocess.STDOUT, text=True)

            # 解析输出并转换为整数
            dimensions = [int(dim) for dim in result.strip().split(',')]
            if len(dimensions) == 2:
                return dimensions[0], dimensions[1]
            else:
                return None, None
        except subprocess.CalledProcessError as e:
            print(f'Error: {e.output}')
            return None, None

    def chatgpt_check_subtitle(self, question):

        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[
                {
                    'role':
                    'system',
                    'content':
                    '用户将输入一段字幕，其中相邻的几句可能是同一句话，但是由于识别错误导致'
                    '了重复。你需要将其中识别错误的语句删去。\n\n例如用户输入：```\n'
                    'Second 61: 你这瓜要是熟我肯定要啊\nSecond 62: 你这瓜要是熟我背定'
                    '要啊\n```\n你应该删去```Second 62: 你这瓜要是熟我背定要啊```，'
                    '保留```Second 61: 你这瓜要是熟我肯定要啊```，所以你的输出应该是：'
                    '\n```Second 61: 你这瓜要是熟我肯定要啊```\n注意：你只需要给出修改'
                    '后的字幕，不要输出任何思考过程。下面是用户输入： '
                },
                {
                    'role': 'user',
                    'content': question
                },
            ],
            temperature=1,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)
        answer = response.choices[0].message.content
        return answer

    def merge_ocr_with_subtitle(self, data):
        if len(data['ocr_subtitle']) == 0:
            return data['merged_subtitle']
        question = '"merged_subtitle": {},\n"ocr_subtitle": {}'.format(
            data['merged_subtitle'], data['ocr_subtitle'])
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[
                {
                    'role':
                    'system',
                    'content':
                    '用户将输入一段语音识别字幕和OCR识别字幕，它们都有识别错误的地方，你需'
                    '要将它们相互校对，形成最符合逻辑的字幕。\n你可以删除几种语音识别字幕中'
                    '的错误：\n1. 中文对话中突然出现英文\n2. 突然出现没有逻辑的语气词，'
                    '如“啊”“吧”等\n3. 不符合语境的词语'
                },
                {
                    'role': 'user',
                    'content': question
                },
            ],
            temperature=1,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)
        answer = response.choices[0].message.content
        second_index = answer.find('Second')
        if second_index != -1:
            result = answer[second_index:]
        else:
            result = answer
        return result.strip('"')

    def find_text_in_dense(self, features, video_width, video_height):
        ocr = features['ocr']
        dense = features['dense_with_pos']
        dense_with_ocr = list()
        ocr_subtitle = list()
        start_time = dense[0]['begin']
        for cur_ocr in ocr:
            time = cur_ocr['begin']
            cur_dense_text = dense[time - start_time]['text']
            cur_dense_with_ocr = defaultdict(str)
            cur_ocr_subtitle = ''
            for cur_text in cur_ocr['text']:
                cur_text_pos = cur_text['pos']
                cur_text_text = cur_text['text']
                if cur_text_pos[1] > video_height * 0.8:
                    cur_ocr_subtitle += cur_text_text + ' '
                    print(f'{cur_text_text} 可能是字幕')
                    continue
                if len(cur_dense_text) == 0:
                    continue
                belong_obj = self.find_pos(cur_text_pos, cur_dense_text)
                if belong_obj:
                    cur_dense_with_ocr[belong_obj] += f'{cur_text_text} '
            if len(cur_dense_with_ocr) > 0:
                final_text = ''
                for key, value in cur_dense_with_ocr.items():
                    final_text += \
                        f'{key} with the words "{value.strip()}" on it, '
                dense_with_ocr.append({
                    'begin': time,
                    'text': final_text.strip(', '),
                })
            if len(cur_ocr_subtitle.strip()):
                ocr_subtitle.append({
                    'begin': time,
                    'text': cur_ocr_subtitle.strip(),
                })
            ocr_subtitle = remove_duplicates_by_text_key(ocr_subtitle, 'text')
        return dense_with_ocr, ocr_subtitle

    def find_pos(self, text_pos, dense_text):
        obj_list = dense_text.split(';')
        min_square = math.inf
        belong_obj = None
        for obj in obj_list:
            obj_item = obj.split(': ')[0]
            if obj_item == ' ':
                continue
            obj_pos = ast.literal_eval(obj.split(': ')[1])
            if obj_pos[0] <= text_pos[0] and \
                    obj_pos[1] <= text_pos[1] and \
                    obj_pos[2] >= text_pos[2] and obj_pos[3] >= \
                    text_pos[3]:
                if compute_square(obj_pos) < min_square:
                    belong_obj = obj_item
                    min_square = compute_square(obj_pos)
        return belong_obj
