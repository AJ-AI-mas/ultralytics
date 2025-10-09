from ultralytics.models import YOLO
import argparse
import os
import shutil

class MyTrainer:

    def initialize_model_from_yaml_or_model(self, model_yaml, model_path):
        if model_yaml == None and model_path == None:
            print("NU AI TRANSMIS CA PARAMETRU NICI MODELUL NICI YAML")
            return None
        is_valid_yaml = (not model_yaml == None) and os.path.isfile(model_yaml) and model_yaml.endswith(".yaml")
        is_valid_path = (not model_path == None) and os.path.isfile(model_path) and model_path.endswith(".pt")
        if (is_valid_yaml and is_valid_path):
            print("SE VA CONTINUA ANTRENAREA SE IGNORA YAML")
            return YOLO(model_path)
        if (is_valid_yaml and not is_valid_path):
            print("SE VA CREEA MODELUL DIN FISIERUL YAML, PATH-UL MODELULUI NU ESTE VALID")
            return YOLO(model_yaml)
        if (not is_valid_yaml and is_valid_path):
            print("SE VA CONTINUA ANTRENAREA SE IGNORA YAML, YAML NU ESTE VALID")
            return YOLO(model_path)
        if (not is_valid_yaml and not is_valid_path):
            print("NU ESTE VALID NICI PATH-UL NICI YAML-UL, cum e posibil")
            return None

    def get_model_filename(self, mode_yaml : str, model_path : str):
        if mode_yaml == None and model_path == None:
            return ""
        if model_path != None:
            model_filename = os.path.basename(model_path)
            return model_filename.replace(".pt", "")
        model_filename = os.path.basename(mode_yaml)
        return model_filename.replace(".yaml", "")

    def train(self, model_yaml, data_yaml, model_path, size):
        model = None
        if (not os.path.isfile(data_yaml)) or (not data_yaml.endswith(".yaml")):
            print("FISIERUL YAML TRANSMIS CA PARAMETRU PENTRU SETUL DE DATE ESTE INVALID")
            exit(1)
        model = self.initialize_model_from_yaml_or_model(model_yaml, model_path)
        if (model == None): 
            print("NU S-A PUTUT INITIALIZA MODELUL\n")
            exit(1)


        destination_location = f"{self.get_model_filename(model_yaml, model_path)}_saved_model"
        if (os.path.isdir(destination_location)):
            shutil.rmtree(destination_location)

        print("SE INCEPE ACUM ANTRENAREA")
        model.train(
            data=data_yaml,
            epochs=2,
            imgsz=size,
            rect=False,
            batch=8,
            save_period=20,
            model=model,
            resume=False,
            verbose=False,
            fraction=0.05,
            project=".",
            name=destination_location,
            patience=50
        )

if "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-yml', '--yaml', help ='yaml file')
    parser.add_argument('-m', '--model', help='model pt path')
    parser.add_argument('-d', '--data', help='data yaml')
    parser.add_argument('-s', '--size', help='image size')
    args = parser.parse_args()

    model_yaml = args.yaml
    data_yaml = args.data
    model_path = args.model
    size = int(args.size)
    trainer = MyTrainer()
    trainer.train(model_yaml, data_yaml, model_path, size)