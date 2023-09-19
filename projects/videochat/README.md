# VideoChat

对视频进行批量QA测试。

# 运行

```shell
# We recommend using conda to manage the environment and use python3.8.16
conda create -n videochat python=3.8.16
conda activate videochat

# Clone the repository:
cd videochat

# Install dependencies:
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install pydantic==1.10.12
apt get ffmpeg

# Download the checkpoints
mkdir pretrained_models
wget -P ./pretrained_models https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth
wget -P ./pretrained_models https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth
wget -P ./pretrained_models https://huggingface.co/Sn4kehead/TransNetV2/resolve/main/transnetv2-pytorch-weights.pth
git clone https://huggingface.co/mrm8488/flan-t5-large-finetuned-openai-summarize_from_feedback ./pretrained_models/flan-t5-large-finetuned-openai-summarize_from_feedback
cd ./pretrained_models/flan-t5-large-finetuned-openai-summarize_from_feedback
git lfs pull
cd ../..

# Configure the necessary ChatGPT APIs
export OPENAI_API_KEY={Your_Private_Openai_Key}

# Run the VideoChat test.
python test.py
```

# 文件结构

- test.py: 代码入口。初始化各个视觉模型与推理的调用，进行问答的测试。
- chatbot.py: 设置ChatGPT的prompt与发送请求。
- configs.ini: 进行各个配置项的设置。
- pretrained_models: 需要下载各个模型的.pth文件，放在这个目录下。
- models: 存放了各个基础模型的class
- transforms.py: 一些数据增强的class
- util.py: 加载视频的函数

# 详细介绍

## configs.ini

- device 使用的gpu
- videos_path 测试视频的目录，其中应该存放若干子目录，子目录中需要包含视频video.mp4和问题data.json
- output_path 问答测试结果存放的目录
- images_path 存放grit模型预测的检测框的可视化结果
- evaluate_path 指定该路径可以对某次问答结果重新评估分数
- appid 调用讯飞api时需要去官网注册appid
- secret_key 调用讯飞api时需要去官网获取secret_key
- segment_length 用于指定internvideo动作预测间隔几秒
- remarks 用于备注本次测试的一些信息
- llm 指定shot summary和问答使用的ChatGPT模型
- predict 是否要进行问答（设置为False时，可以结合evaluate_path评估特定路径下的结果）
- evaluate 是否要对问答结果评估分数
- mode 如果是normal，则不进行分段；如果是shot，则进行分段

## pretrained_models

需要包含以下目录和文件

- flan-t5-large-finetuned-openai-summarize_from_feedback 目录
- uniformerv2 目录
- grit_b_densecap_objectdet.pth
- tag2text_swin_14m.pth
- transnetv2-pytorch-weights.pth

## test.py

包含3个class

### InputVideo

1. init：从videos_path下读取相应的文件，如果features文件不存在，会调用self.extract_features进行特征提取，并保存到视频所在目录
2. self.extract_features：首先调用self.video_chat.inference_second进行特征的提取，随后会调用ProcessSubtitle将whisper和讯飞的结果进行合并，调用ShotProcessor处理shot分段的结果（帧->秒），它们处理完的结果都会保存到features.json中
3. start_test：首先从features.json中读取特征，然后根据mode来进行不分段的推理或shot的推理。shot推理依赖于features\['time_intervals'\]的结果（参考models/transnetv2.py），将features的内容按照shot的时间段进行划分，每一段shot对应的特征进入ChatGPT进行summary。

### VideoChat

1. load_model：加载基础模型
2. inference_second：将视频每秒抽取一帧，提取特征，所有特征都是一个list，list中包含{'begin': int, 'text': str}类型的对象。这里提取的特征包括whisper（没有说话人的字幕）、讯飞subtitle（有说话人的字幕）、dense（画面中的物体描述）、dense_with_pos（dense的检测框）、frame caption（画面的描述）、shot（每个shot的起始帧和结束帧）、synth_caption（根据frame形成的总结性描述）、tag（视频中出现的物体类别）

|      特征       |         模型         |    在哪里调用    |
| :-------------: | :------------------: | :--------------: |
|      shot       | models/transnetv2.py | inference_second |
|    subtitle     |  models/subtitle.py  | inference_second |
|     whisper     |       whisper        | inference_second |
|      dense      | models/grit_model.py | inference_second |
| dense_with_pos  | models/grit_model.py | inference_second |
|      frame      |  models/tag2text.py  | inference_second |
|  synth_caption  |       simplet5       | inference_second |
| time_intervals  | models/transnetv2.py | extract_features |
| merged_subtitle |  models/subtitle.py  | extract_features |
|       ocr       |    models/ocr.py     | inference_second |
| dense_with_ocr  |    models/ocr.py     | extract_features |
| final_subtitle  |    models/ocr.py     | extract_features |

## chatbot.py

用于构造prompt和向chatgpt发送请求

### ConversationBot

1. init_agent：用于在非shot情况下的prompt构造，输入是全部features
2. init_agent_shot：用于在shot情况下的prompt构造，输入是shot features
3. init_agent_with_summary：在所有shot转summary后，根据summary构造prompt，输入是summary list
4. run_text：用于发送请求进行问答

### ChainOfThought

使用LangChain来构造顺序链，以及规范ChatGPT输出

## models/subtitle.py

包含了两个class，用于调用讯飞api，以及将讯飞api的结果和whisper合并

### RequestApi

- 在test.py的extract_features中调用，提取features\['subtitle'\]

1. upload：通过该函数对注册好的语音转写服务发送请求，在响应体中能获得一个orderId，通过这个orderId可以请求到转写的结果
2. get_result：根据orderId获取转写的结果
3. result2text：由于转写结果是以分词的形式返回的，这里将其处理成{'begin': int, 'end': int, 'speaker': int, 'text': str}形式的对象

### ProcessSubtitle

- 在test.py的features提取完成后调用，提取features\['merged_subtitle'\]

1. merge_whisper_and_xunfei：由于讯飞对中文的识别能力较强且有说话人识别，whisper对外语的识别能力较强，这里的逻辑是如果whisper判断为非中文，就采用whisper的结果，并根据这一句的begin时间查找时间最近的讯飞中的句子的speaker。如果whisper判断为中文，直接查找时间最近的讯飞中的对象。
2. find_match_subtitle：查找begin时间最近的对象
3. remove_duplicates_by_text_key：由于whisper和讯飞的分句不同，上述结果可能会将一句话append两次，这里删去重复的语句

## models/transnetv2.py

包含了计算视频分段，以及将分段帧转时间的全部方法

### Shot

- 在test.py的extract_features中调用，提取features\['shot'\]

1. init：加载模型
2. inference： 获取每一段shot的起始帧和结束帧

### ShotProcessor

- 在test.py的features提取完成后调用，提取features\['time_intervals'\]

1. shot：根据视频帧率，将shot的起始帧和结束帧都转化为时间

## models/ocr.py

使用paddleocr进行文字提取，暂时无法兼容

### ProcessOCR

1. inference: 使用paddleocr获取画面中文字的内容和位置，保存到features\['ocr'\]
2. merge：将ocr结果和features\['merged_subtitle'\]合并。
   根据ocr出现的位置，如果位置在画面下方20%，则认为是字幕，由于一般的规则无法处理ocr字幕和语音字幕的合并，这里调用GPT4来实现。
3. find_text_in_dense：如果判断ocr文字不是字幕，则将其和features\['dense'\]的结果进行比对，查找dense中bbox包含该ocr的bbox的最小物体，将文字内容和找到的物体进行合并。

# 引用

The project is based on

- [Ask-Anything](https://github.com/OpenGVLab/Ask-Anything/)
- [InternVideo](https://github.com/OpenGVLab/InternVideo)
- [Tag2Text](https://github.com/xinyu1205/Tag2Text)
- [GRiT](https://github.com/JialianW/GRiT)
- [mrm8488](https://huggingface.co/mrm8488/flan-t5-large-finetuned-openai-summarize_from_feedback)
- [ChatGPT](https://openai.com/blog/chatgpt)
- [TransNetV2](https://github.com/soCzech/TransNetV2)
- [LangChain](https://github.com/langchain-ai/langchain)

Thanks for the authors for their efforts.
