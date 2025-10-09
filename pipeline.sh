#! /bin/bash

# https://github.com/google-coral/webcoral.git
set -e
YAML=$1
SIZE=1024

DATASET_YAML=ultralytics/cfg/datasets/VisDrone.yaml
# DATASET_YAML=ultralytics/cfg/datasets/downtest.yaml
# DATASET_YAML=ultralytics/cfg/datasets/coco.yaml
# DATASET_YAML=ultralytics/cfg/datasets/mystandford.yaml

MODEL_NAME=$(basename $YAML .yaml)
SAVE_MODEL_PATH=${MODEL_NAME}_saved_model
MODEL=${MODEL_NAME}.pt

# export PYENV_ROOT="$HOME/.pyenv"
# export PATH="$PYENV_ROOT/bin:$PATH"
# eval "$(pyenv init --path)"
# eval "$(pyenv init -)"
# pyenv shell 3.11.1

source venv/bin/activate
if [ ! -d ${SAVE_MODEL_PATH} ]
then
    mkdir ${SAVE_MODEL_PATH}
fi

python3 train.py \
--size ${SIZE} \
--data ${DATASET_YAML} \
--yaml ${YAML} \
--model ${MODEL} 

mv ${SAVE_MODEL_PATH}/weights/best.pt ${SAVE_MODEL_PATH}/${MODEL}
mv ${SAVE_MODEL_PATH}/weights/last.pt ${SAVE_MODEL_PATH}
rm -r ${SAVE_MODEL_PATH}/weights
rm -r ${SAVE_MODEL_PATH}/*.jpg ${SAVE_MODEL_PATH}/*.png


python3 archiver.py --folder ${SAVE_MODEL_PATH}

source venv/bin/activate
python3 visualize.py \
--data "./datasets/downtest/" \
--labels "./res/${MODEL_NAME}"
deactivate
