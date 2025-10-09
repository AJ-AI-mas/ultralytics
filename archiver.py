import argparse
import os
import yaml
from zipfile import ZipFile

class Archiver:
    def read_yaml_file_extract_data(self, yaml_file : str):
        values = {}
        with open(yaml_file, "r") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
            values["epochs"] = data["epochs"]
            values["batch"] = data["batch"]

            data_path = data["data"]
            dataset_name = os.path.basename(data_path)
            values["dataset"] = dataset_name.replace(".yaml", "")
        return values


    def archive(self, folder_path : str):
        filenames = os.listdir(folder_path)
        if not "args.yaml" in filenames:
            print("The file with model parameters is no longer there.")
            exit(1)

        info = self.read_yaml_file_extract_data(os.path.join(folder_path, "args.yaml"))
        zip_filename = f"archives/{folder_path}_{info["dataset"]}_{info["epochs"]}_batch{info["batch"]}.zip"
        filepaths = [os.path.join(folder_path, filename) for filename in filenames] 
        with ZipFile(zip_filename, 'w') as zip:
            for file in filepaths:
                zip.write(file)
        

if "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help ='model folder to archive')
    args = parser.parse_args()
    folder_path = args.folder
    if (folder_path == None):
        print("FOLDER PATH IS MISSING")
        exit(1)
    
    archiver = Archiver()
    archiver.archive(folder_path)
