import json
from os import path


def update_json(maps_dir):    
    maps_json = path.join(maps_dir, "maps.json")
    with open(maps_json, "r") as f:
        parameters_dict = json.load(f)

    parameters_dict["split"] = [0, 1, 2, 3, 4, 5]
    json_data = json.dumps(parameters_dict, skipkeys=True, indent=4)
    with open(maps_json, "w") as f:
        f.write(json_data)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('maps_dir')
    args = parser.parse_args()

    update_json(args.maps_dir)
