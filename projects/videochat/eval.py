import json
import os
import time

import openai


def get_answer(question, answer_list):
    try:
        answer = '\n'.join(answer_list)
        query = f'''问题:"{question}"\n回答列表:\n"{answer}"'''
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[{
                'role':
                'system',
                'content':
                '假设有一个视频，用户从其中提取了若干帧图像，并对每一帧图像都提问了相同的问'
                '题，并有一个视觉模型做出了相应的回答。但实际上该问题只与其中的部分图像相关'
                '，因此你需要从其中筛选并整合出比较合理的回答。以如下json格式回答'
                "{'answer':string, 'reason':string}"
            }, {
                'role': 'user',
                'content': query,
            }],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)
        answer = response.choices[0].message.content
        answer = answer.strip('{}').replace("\"", "'")
        result = dict()
        result['answer'] = answer.split(
            ', \'reason\':')[0][len('\'answer\': '):].strip('\'')
        print(answer)
    except openai.error.RateLimitError as e:
        print(e)
        time.sleep(60)  # Wait for 1 minute before retrying
        return get_answer(question, answer_list)
    return result['answer']


def main():
    main_path = '/mnt/data.coronaryct.1/ZhuYichen/videochat/output' \
                '/20230915014800 '
    for folder in os.listdir(main_path):
        file_path = os.path.join(
            os.path.join(main_path, folder), 'output.json')
        new_file_path = os.path.join(
            os.path.join(main_path, folder), 'output_final.json')
        with open(file_path) as file:
            data = json.load(file)
            for i, qa in enumerate(data['qa']):
                answer = get_answer(qa['q'], qa['predict'])
                data['qa'][i]['predict'] = answer
            with open(new_file_path, 'w') as json_file:
                json.dump(data, json_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()
