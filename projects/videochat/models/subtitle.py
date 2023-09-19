import base64
import hashlib
import hmac
import json
import math
import os
import time
import urllib

import requests


class RequestApi(object):

    def __init__(self, appid, secret_key, upload_file_path):
        self.lfasr_host = 'https://raasr.xfyun.cn/v2/api'
        self.api_upload = '/upload'
        self.api_get_result = '/getResult'
        self.result = None
        self.appid = appid
        self.secret_key = secret_key
        self.upload_file_path = upload_file_path
        self.ts = str(int(time.time()))
        self.signa = self.get_signa()

    def get_signa(self):
        appid = self.appid
        secret_key = self.secret_key
        m2 = hashlib.md5()
        m2.update((appid + self.ts).encode('utf-8'))
        md5 = m2.hexdigest()
        md5 = bytes(md5, encoding='utf-8')
        # 以secret_key为key, 上面的md5为msg， 使用hashlib.sha1加密结果为signa
        signa = hmac.new(secret_key.encode('utf-8'), md5,
                         hashlib.sha1).digest()
        signa = base64.b64encode(signa)
        signa = str(signa, 'utf-8')
        return signa

    def upload(self):
        print('上传部分：')
        upload_file_path = self.upload_file_path
        file_len = os.path.getsize(upload_file_path)
        file_name = os.path.basename(upload_file_path)

        param_dict = {}
        param_dict['appId'] = self.appid
        param_dict['signa'] = self.signa
        param_dict['ts'] = self.ts
        param_dict['fileSize'] = file_len
        param_dict['fileName'] = file_name
        param_dict['duration'] = '200'
        param_dict['roleType'] = 1
        # param_dict["roleNum"] = 2
        print('upload参数：', param_dict)
        data = open(upload_file_path, 'rb').read(file_len)

        response = requests.post(
            url=self.lfasr_host + self.api_upload + '?' +
            urllib.parse.urlencode(param_dict),
            headers={'Content-type': 'application/json'},
            data=data)
        print('upload_url:', response.request.url)
        result = json.loads(response.text)
        print('upload resp:', result)
        return result

    def get_result(self):
        uploadresp = self.upload()
        orderId = uploadresp['content']['orderId']
        param_dict = {
            'appId': self.appid,
            'signa': self.signa,
            'ts': self.ts,
            'orderId': orderId,
            'resultType': 'transfer'
        }
        print('')
        print('查询部分：')
        print('get result参数：', param_dict)
        status = 3
        # 建议使用回调的方式查询结果，查询接口有请求频率限制
        while status == 3:
            response = requests.post(
                url=self.lfasr_host + self.api_get_result + '?' +
                urllib.parse.urlencode(param_dict),
                headers={'Content-type': 'application/json'})
            # print("get_result_url:",response.request.url)
            result = json.loads(response.text)
            print(result)
            status = result['content']['orderInfo']['status']
            print('status=', status)
            if status == 4:
                break
            time.sleep(5)
        print('get_result resp:', result)
        self.result = result
        return result

    def result2text(self, return_list=False):
        order_result = json.loads(self.result['content']['orderResult'])
        lattice = order_result['lattice']
        sentence_list = list()
        for item in lattice:
            sentence = dict()
            text = ''
            for item2 in json.loads(item['json_1best'])['st']['rt'][0]['ws']:
                text += item2['cw'][0]['w']
            sentence['text'] = text
            sentence['speaker'] = json.loads(item['json_1best'])['st']['rl']
            sentence['begin'] = int(
                int(json.loads(item['json_1best'])['st']['bg']) / 1000)
            sentence['end'] = int(
                int(json.loads(item['json_1best'])['st']['ed']) / 1000)
            sentence_list.append(sentence)
        result = ''
        for sentence in sentence_list:
            result += 'Second {} to Second {}, Speaker {}: {}\n'.format(
                sentence['begin'], sentence['end'], sentence['speaker'],
                sentence['text'])
        if return_list:
            return sentence_list
        else:
            return result


class ProcessSubtitle:

    def __init__(self, features):
        self.features = features

    def merge_whisper_and_xunfei(self):
        whisper = self.features['whisper']
        xunfei = self.features['subtitle']
        result = []
        for whisper_item in whisper:
            # 第一个字是中文，使用讯飞的结果
            if '\u4e00' <= whisper_item['text'][0] <= '\u9fff':
                find_item = self.find_match_subtitle(whisper_item['begin'],
                                                     xunfei)
                if not find_item:
                    result.append(whisper_item)
                else:
                    result.append(find_item)
            # 不是中文，使用whisper的结果，但使用讯飞的speaker
            else:
                find_item = self.find_match_subtitle(whisper_item['begin'],
                                                     xunfei)
                if not find_item:
                    whisper_item['speaker'] = '1'
                    result.append(whisper_item)
                else:
                    whisper_item['speaker'] = find_item['speaker']
                    result.append(whisper_item)
        result = self.remove_duplicates_by_text_key(result, 'text')
        return result

    def find_match_subtitle(self, time, subtitle):
        time_gap = math.inf
        temp = None
        for item in subtitle:
            if item['begin'] == time:
                return item
            elif 0 < time - item['begin'] < time_gap:
                time_gap = time - item['begin']
                temp = item
            elif time - item['begin'] < 0:
                if item['begin'] - time < time_gap:
                    return item
                else:
                    return temp
        return temp

    def remove_duplicates_by_text_key(self, lst, value):
        seen_text_values = set()
        result = []

        for obj in lst:
            text_value = obj.get(value)

            if text_value not in seen_text_values:
                seen_text_values.add(text_value)
                result.append(obj)

        return result


# 输入讯飞开放平台的appid，secret_key和待转写的文件路径
if __name__ == '__main__':
    api = RequestApi(appid='', secret_key='', upload_file_path=r'')

    api.get_result()
    api.result2text()
