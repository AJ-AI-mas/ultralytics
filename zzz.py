import fiftyone as fo
from ultralytics import YOLO

dataset = fo.Dataset.from_dir(
    dataset_dir="./datasets/VisDrone",
    dataset_type=fo.types.YOLOv5Dataset,
    name="VisDrone"
)

model=YOLO("yolo11n.pt")

dataset.apply_model(
    model,
    label_field="predictions",
    confidence_thresh=0.25,   
    batch_size=8,
    num_workers=4,
)

results = dataset.evaluate_detections(
    "predictions",        # predicted field
    gt_field="ground_truth",
    eval_key="eval",      # evaluation run key stored in dataset
    iou=0.5,              # IoU threshold (change if desired)
    method="coco"         # or "open-images" etc.
)
results.print_report()
fo.pprint(dataset.stats(include_media=True))
session = fo.launch_app(dataset)
session.wait()
