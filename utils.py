import json


def save_json_file(obj, save_file):
    """存储语料库"""
    with open(save_file, "w") as f:
        json.dump(obj, f)


def load_json_file(file_path):
    """读取语料库"""
    with open(file_path, "r") as f:
        return json.load(f)