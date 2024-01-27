# ancient-chat-llm 古语说

## 演示

## 前言

古语说 - 一个精通中国文化的大模型

## 知识库

[x] 成语
[x] 论语
[x] 唐诗
[x] 宋词
[x] 楚辞
[x] 四书五经
[ ] 史记
[ ] 宫廷制度
[ ] 二十四节气

## 数据集

目前使用到的开源数据集有以下几个：

- 文言文：https://huggingface.co/datasets/RUCAIBox/Erya-dataset/tree/main
- 古诗：https://github.com/chinese-poetry/chinese-poetry

数据集结构（省略了用不到的文件）：

```bash
dataset/
├── Erya-dataset
│   ├── dataset # 解压自 finetune.tgz 
│   └── stage_2 # 解压自 trans.tgz 
└── chinese-poetry
    ├── 五代诗词
    ├── 元曲
    ├── 全唐诗
    ├── 四书五经
    ├── 宋词
    ├── 幽梦影
    ├── 御定全唐詩
    ├── 曹操诗集
    ├── 楚辞
    ├── 水墨唐诗
    ├── 纳兰性德
    ├── 蒙学
    ├── 论语
    └── 诗经

```

使用脚本可以进行生成：

```bash
cd dataset
python gen_dataset.py --data=dataset --output=data.jsonl
```

## 搭建环境

本项目使用 [xtuner](https://github.com/InternLM/xtuner) 训练，在 [internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b) 上进行微调

训练之前，需要在 xtuner 代码中 `xtuner/xtuner/utils/templates.py` 添加 `SYSTEM_TEMPLATE.chinese_old_saying` ：

```python
SYSTEM_TEMPLATE = ConfigDict(
    moss_sft=('You are an AI assistant whose name is {bot_name}.\n'
              'Capabilities and tools that {bot_name} can possess.\n'
              '- Inner thoughts: enabled.\n'
              '- Web search: enabled. API: Search(query)\n'
              '- Calculator: enabled. API: Calculate(expression)\n'
              '- Equation solver: enabled. API: Solve(equation)\n'
              '- Text-to-image: disabled.\n'
              '- Image edition: disabled.\n'
              '- Text-to-speech: disabled.\n'),
    alpaca=('Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.\n'),
    arxiv_gentile=('If you are an expert in writing papers, please generate '
                   "a good paper title for this paper based on other authors' "
                   'descriptions of their abstracts.\n'),
    colorist=('You are a professional color designer. Please provide the '
              'corresponding colors based on the description of Human.\n'),
    coder=('You are a professional programer. Please provide the '
           'corresponding code based on the description of Human.\n'),
    lawyer='你现在是一名专业的中国律师，请根据用户的问题给出准确、有理有据的回复。\n',
    medical='如果你是一名医生，请根据患者的描述回答医学问题。\n',
    sql=('If you are an expert in SQL, please generate a good SQL Query '
         'for Question based on the CREATE TABLE statement.\n'),
+    chinese_old_saying="你是一位专业的中文教师。你总能解答用户关于中文的相关知识。\n",
)
```

## 训练

修改数据集路径，以及模型路径

```python
# Model
- pretrained_model_name_or_path = 'internlm/internlm2-7b'
+ pretrained_model_name_or_path = '/path/to/internlm/internlm2-7b' # 这步可选，如果事先下载好了模型可以直接使用绝对路径

# Data
- data_path = 'timdettmers/openassistant-guanaco'
+ data_path = '/path/to/data.jsonl' # 数据集步骤生成的 json 文件绝对路径
prompt_template = PROMPT_TEMPLATE.default
max_length = 2048
pack_to_max_length = True

```

使用命令进行训练：

```bash
xtuner train finetune_configs/internlm2_chat_7b/internlm2_chat_7b_qlora_custom_data_e3_finetune.py --deepspeed deepspeed_zero2
```

注意：如果显存不够了，调小一点 `batch_size` 和 `max_length`，反之还剩很多，调大这两个值

## 部署

## 后记

本模型在数据集方面的还没做很精细的调优，还有很多不足的地方，大家可以一起讨论，如果大家有数据集，可以再 issue 留言讨论