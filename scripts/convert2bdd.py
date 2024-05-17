from pathlib import Path
import sys
import argparse
from glob import glob
import os
from pprint import pprint
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import load_json, load_text, dump_json


def run(anno_root):
    bdd_results = {"frame_list": []}
    class_idxs = load_text("./config/ids.txt")
    train_anno_paths = glob(os.path.join(anno_root, "train", "*.json"))
    val_anno_paths = glob(os.path.join(anno_root, "val", "*.json"))
    anno_paths = list(train_anno_paths) + list(val_anno_paths)
    for anno_path in tqdm(anno_paths):
        name = anno_path.split("/")
        name = name[-1].replace(".json", ".jpg")
        annos = load_json(anno_path)
        for frame in annos["frames"]:
            tmp = []
            for obj in frame["objects"]:
                category = obj["category"]
                if category not in class_idxs:
                    continue
                tmp.append(obj)
            tmp_infos = {"labels": tmp, "attributes": annos["attributes"]}
        tmp_infos["name"] = name
        tmp_infos["dataset"] = "bdd"
        bdd_results["frame_list"].append(tmp_infos)
    dump_json(path=os.path.join("/aidata/anders/autosys/annos",
                                "BDD_100K.json"),
              data=bdd_results)
    print("-" * 100)
    print("INFO: Dump bdd json... ")


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_root', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    run(args.anno_root)
