import sys
import os
from os import path
import json
from clinicadl import MapsManager


def get_info_from_log(log):
    with open(log) as f:
        txt = f.read()
        for line in txt.split('\n'):
            if "Split list: " in line:
                splits = set()
                for car in line:
                    if car.isnumeric():
                        splits.add(int(car))
            if "A new MAPS was created" in line:
                i = 1
                while line[-i].isnumeric():
                    i+=1
                n_maps = int(line[-i+1:])
            if "Successfully launched training !" in line:
                break
    return n_maps, splits


def get_trained_split(maps_dir):
    trained_splits = set()
    for sub_folder in os.listdir(maps_dir):
        if "split" in sub_folder:
            model_path = path.join(maps_dir, sub_folder, "best-loss", 'model.pt')
            if path.exists(model_path):
                trained_splits.add(int(sub_folder[-1]))
    return trained_splits


def get_parameters(maps_dir):
    json_path = path.join(maps_dir, "maps.json")
    with open(json_path, "r") as f:
        parameters_dict = json.load(f)
    return parameters_dict


model = sys.argv[1]
log = sys.argv[2]

n_maps, splits = get_info_from_log(log)

model_rs_dir = path.dirname(path.dirname(log))
maps_dir = path.join(model_rs_dir, 'maps', f"MAPS_{model}_{n_maps}")
print("MAPS directory:", maps_dir)
trained_splits = get_trained_split(maps_dir)
splits_to_train = list(splits - trained_splits)
print("Split to train:", splits_to_train)

parameters = get_parameters(maps_dir)
print("Parameters:", parameters)

new_maps_dir = maps_dir + "_resume"
maps_manager = MapsManager(new_maps_dir, parameters, verbose="info")
maps_manager.train_pythae(splits_to_train)
