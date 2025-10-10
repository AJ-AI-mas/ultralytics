# python3 validate.py --size 1024 --model yolov5C2Ghost.pt --data ultralytics/cfg/datasets/mystandford.yaml
from ultralytics.models import YOLO
import argparse
import os
# python3 validate.py --model yolo11n.pt --data ultralytics/cfg/datasets/VisDrone.yaml --size 1024

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model pt path')
parser.add_argument('-d', '--data', help='data yaml')
parser.add_argument('-s', '--size', help='image size')
args = parser.parse_args()

data_yaml = args.data
model_path = args.model
size = int(args.size)
model = None

def initialize_model_from_yaml_or_model(model_path):
    if model_path == None:
        print("NU AI TRANSMIS CA PARAMETRU NICI MODELUL NICI YAML")
        return None
    is_valid_path = (not model_path == None) and os.path.isfile(model_path) and model_path.endswith(".pt")
    if is_valid_path:
        return YOLO(model_path)
    return None 
    
if (not os.path.isfile(data_yaml)) or (not data_yaml.endswith(".yaml")):
    print("FISIERUL YAML TRANSMIS CA PARAMETRU PENTRU SETUL DE DATE ESTE INVALID")
    exit(1)
model = initialize_model_from_yaml_or_model(model_path)
if (model == None): 
    print("NU S-A PUTUT INITIALIZA MODELUL\n")
    exit(1)

print("SE INCEPE ACUM VALIDAREA")
results = model.val(
    data=data_yaml,
    imgsz=size,
    conf=0.25,
    iou=0.3,
    rect=False,
    batch=8,
    save_conf=True,
    workers=4,
    verbose=True,
    save_json=True,
    save_txt=True
)

print(results.confusion_matrix.to_df())
