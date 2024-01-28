#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024.1.28
# @Author  : HinGwenWong

"""本脚本用于生成数据集

数据集链接：
- 文言文：https://huggingface.co/datasets/RUCAIBox/Erya-dataset/tree/main
- 古诗：https://github.com/chinese-poetry/chinese-poetry

请根据进行数据集准备:

dataset/
├── Erya-dataset
│   ├── dataset
│   └── stage_2
├── chinese-poetry
│   ├── 五代诗词
│   ├── 元曲
│   ├── 全唐诗
│   ├── 四书五经
│   ├── 宋词
│   ├── 幽梦影
│   ├── 御定全唐詩
│   ├── 曹操诗集
│   ├── 楚辞
│   ├── 水墨唐诗
│   ├── 纳兰性德
│   ├── 蒙学
│   ├── 论语
│   └── 诗经
├── gen_dataset.py
└── whoami.jsonl

执行脚本生成：
`python gen_dataset.py --data-root=./dataset --output=data.jsonl`

"""
import json
from pathlib import Path
import argparse
import zhconv


def get_erya_data(data_root: Path, dir_name_list: list, json_save_path: Path):
    """将 Erya 数据集格式化为训练集

    Args:
        data_root (Path): 数据集根目录
        dir_name_list (list): 需要遍历的文件夹名称
        json_save_path (Path): 生成数据集 json 文件路径
    """
    qa_list = []
    file_path_list = []
    for dir_name in dir_name_list:
        data_path = data_root.joinpath(dir_name)
        for sub_dir in data_path.iterdir():
            if sub_dir.is_file() and sub_dir.suffix == ".src":
                file_path_list.append(sub_dir)
            elif sub_dir.is_dir():
                for sub_file in sub_dir.iterdir():
                    if sub_file.is_file() and sub_file.suffix == ".src":
                        file_path_list.append(sub_file)

    for src_file in file_path_list:
        print(f"Processing {src_file}")
        with open(src_file, "r", encoding="utf-8") as f_src:
            src_lines = f_src.readlines()

        with open(src_file.with_suffix(".tgt"), "r", encoding="utf-8") as f_tgt:
            tgt_lines = f_tgt.readlines()

        assert len(src_lines) == len(tgt_lines)

        print(f"Total len = {len(qa_list)}")
        for idx in range(len(src_lines)):
            qa_list_tmp = []
            qa_list_tmp.append(
                {"Q": f"帮我翻译成白话文：{src_lines[idx]}", "A": f"{tgt_lines[idx]}"}
            )

            qa_list_tmp.append(
                {"Q": f"帮我翻译成文言文：{tgt_lines[idx]}", "A": f"{src_lines[idx]}"}
            )

            for qa in qa_list_tmp:
                qa_list.append(
                    {
                        "conversation": [
                            {
                                "system": SYSTE_STR,
                                "input": qa["Q"],
                                "output": qa["A"],
                            }
                        ],
                    }
                )

    print("Saving")
    with open(json_save_path, "w", encoding="utf-8") as f_output:
        f_output.write(json.dumps(qa_list, ensure_ascii=False, indent=4))
    print("Save done")


def get_poetry_data(data_root: Path, json_save_path: Path):
    """将古诗词数据集格式化为训练集

    Args:
        data_root (Path): 数据集根目录
        json_save_path (Path): 生成数据集 json 文件路径
    """

    # 每个子集有独立的字段映射
    dir_name_info_dict = {
        "论语": {"title": "chapter", "paragraphs": "paragraphs", "author": "孔子"},
        "蒙学": {"title": "title", "paragraphs": "paragraphs", "author": "<author>"},
        "曹操诗集": {"title": "title", "paragraphs": "paragraphs", "author": "曹操"},
        "楚辞": {"title": "title", "paragraphs": "content", "author": "<author>"},
        "全唐诗": {"title": "title", "paragraphs": "paragraphs", "author": "<author>"},
        "诗经": {"title": "title", "paragraphs": "content", "author": "诗经"},
        "宋词": {"title": "rhythmic", "paragraphs": "paragraphs", "author": "<author>"},
        "御定全唐詩": {"title": "title", "paragraphs": "paragraphs", "author": "<author>"},
        "五代诗词": {"title": "title", "paragraphs": "paragraphs", "author": "<author>"},
        "元曲": {"title": "title", "paragraphs": "paragraphs", "author": "<author>"},
        "水墨唐诗": {"title": "title", "paragraphs": "paragraphs", "author": "<author>"},
        "纳兰性德": {"title": "title", "paragraphs": "para", "author": "纳兰性德"},
        "四书五经": {"title": "chapter", "paragraphs": "paragraphs", "author": "孔子"},
        # "幽梦影",
    }

    normal_list = [
        "曹操诗集/caocao.json",
        "楚辞/chuci.json",
        "诗经/shijing.json",
        "全唐诗/唐诗三百首.json",
        "宋词/宋词三百首.json",
        "蒙学/baijiaxing.json",
        "蒙学/sanzijing-new.json",
        "蒙学/qianziwen.json",
        "蒙学/zhuzijiaxun.json",
    ]

    author_list = [
        "全唐诗/authors.song.json",
        "全唐诗/authors.tang.json",
        "宋词/author.song.json",
    ]

    qa_list = []

    all_author_info = dict()
    # 作者介绍
    for author_path in author_list:
        with open(data_root.joinpath(author_path), "r", encoding="utf-8") as f_author:
            author_json = json.load(f_author)

            for author_info in author_json:
                desc_feild_name = "desc"
                if desc_feild_name not in author_info.keys():
                    desc_feild_name = "description"

                if author_info[desc_feild_name] == "--":
                    continue

                author_info["name"] = zhconv.convert(author_info["name"], "zh-hans")

                author_info[desc_feild_name] = (
                    author_info["name"] + "," + author_info[desc_feild_name][2:]
                )  # 去掉前面的 --

                qa_list_tmp = []
                qa_list_tmp.append(
                    {
                        "Q": f"介绍下{author_info['name']}",
                        "A": f"{author_info[desc_feild_name]}",
                    }
                )
                qa_list_tmp.append(
                    {
                        "Q": f"{author_info['name']}是谁",
                        "A": f"{author_info[desc_feild_name]}",
                    }
                )
                qa_list_tmp.append(
                    {
                        "Q": f"{author_info['name']}简介",
                        "A": f"{author_info[desc_feild_name]}",
                    }
                )

                all_author_info.update(
                    {author_info["name"]: author_info[desc_feild_name]}
                )
                for qa in qa_list_tmp:
                    qa_list.append(
                        {
                            "conversation": [
                                {
                                    "system": SYSTE_STR,
                                    "input": qa["Q"],
                                    "output": qa["A"],
                                }
                            ],
                        }
                    )
    # 古诗内容
    for book_path in normal_list:
        with open(data_root.joinpath(book_path), "r", encoding="utf-8") as f_book:
            book_json = json.load(f_book)
            if not isinstance(book_json, list):
                book_json = [book_json]

            dir_name_info = dir_name_info_dict[Path(book_path).parent.name]

            for poetry in book_json:
                qa_list_tmp = []
                title = zhconv.convert(poetry[dir_name_info["title"]], "zh-hans")
                paragraphs = "\n".join(poetry[dir_name_info["paragraphs"]])
                qa_list_tmp.append(
                    {
                        "Q": f"背诵《{title}》",
                        "A": f"{paragraphs}",
                    }
                )

                if "translate" in poetry and len(poetry["translate"]) > 0:
                    translate = "\n".join(poetry["translate"])
                    qa_list_tmp.append(
                        {
                            "Q": f"翻译《{title}》",
                            "A": f"{translate}",
                        }
                    )

                if "appreciation" in poetry and len(poetry["appreciation"]) > 0:
                    appreciation = "\n".join(poetry["appreciation"])
                    qa_list_tmp.append(
                        {
                            "Q": f"介绍下《{title}》",
                            "A": f"{appreciation}",
                        }
                    )

                if (
                    dir_name_info["author"] in poetry
                    and poetry[dir_name_info["author"]] == "<author>"
                ):
                    author = zhconv.convert(poetry[dir_name_info["author"]], "zh-hans")
                else:
                    author = dir_name_info["author"]
                author += "写的。"
                if all_author_info.get(author) is not None:
                    author += all_author_info.get(author)
                qa_list_tmp.append(
                    {
                        "Q": f"《{title}》谁写的",
                        "A": f"《{title}》是由{author}",
                    }
                )

                for qa in qa_list_tmp:
                    qa_list.append(
                        {
                            "conversation": [
                                {
                                    "system": SYSTE_STR,
                                    "input": qa["Q"],
                                    "output": qa["A"],
                                }
                            ],
                        }
                    )

    print("Saving")
    with open(json_save_path, "w", encoding="utf-8") as f_output:
        f_output.write(json.dumps(qa_list, ensure_ascii=False, indent=4))
    print("Save done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gen dataset")
    parser.add_argument("--data-root", required=True, help="Root path of dataset.")
    parser.add_argument(
        "--output", default="data.jsonl", help="Save train dataset json path."
    )
    args = parser.parse_args()

    DATA_ROOT = Path(args.data_root)

    if DATA_ROOT.exists():
        raise FileNotFoundError(f"Can't not found data root {DATA_ROOT}")

    SYSTE_STR = "你是一位专业的中文教师。你总能解答用户关于中文的相关知识。"
    
    # 文言文白话文数据集
    DATA_ROOT_ERYA = DATA_ROOT.joinpath(r"./Erya-dataset")
    get_erya_data(
        DATA_ROOT_ERYA,
        dir_name_list=["dataset", "stage_2"],
        json_save_path=DATA_ROOT_ERYA.joinpath("data-erya.jsonl"),
    )

    # 古诗数据集
    DATA_ROOT_POETRY = DATA_ROOT.joinpath(r"./chinese-poetry")
    get_poetry_data(
        DATA_ROOT_POETRY, json_save_path=DATA_ROOT_POETRY.joinpath("data-poetry.jsonl")
    )

    # merge
    new_json = []
    for json_file in [
        DATA_ROOT_ERYA.joinpath("data-erya.jsonl"),
        DATA_ROOT_POETRY.joinpath("data-poetry.jsonl"),
        DATA_ROOT.joinpath("whoami.jsonl")
    ]:
        with open(json_file, "r", encoding="utf-8") as f_json:
            json_data = json.load(f_json)
            print(f"len for {json_file} is {len(json_data)}")
            new_json += json_data

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    print(f"Total dataset len = {len(new_json)}")
    with open(args.output, "w", encoding="utf-8") as f_output:
        f_output.write(json.dumps(new_json, ensure_ascii=False, indent=4) + "\n")

    print(f"All done, save dataset jsonl to {output_path}")
