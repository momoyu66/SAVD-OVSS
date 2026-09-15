# Data and generated caches

Datasets and generated feature caches are not tracked. The paper experiments
used the following repository-relative layout:

    data/coco/annotations/captions_train2017.json
    data/coco/train2017/
    data/flow_distill_vadd/coco_train_2k_64anchors_fp16.pth
    data/pascal_voc_processed/
    data/pascal_context_processed/
    data/coco_object_processed/
    data/coco_stuff_processed/
    data/cityscapes_processed/
    data/ade20k_processed/

Large datasets may be linked from external storage rather than copied.
Preprocessing utilities are under processing/.

The reported visual-anchor cache has:

- 2,000 COCO training images;
- 64 normalized DINOv3 patch features per image;
- shape 128000 x 1024;
- fp16 storage;
- SHA-256
  395c3b7b0e7ef4d53d8955017b47695e7a0961db933b3c78916ce27b1e535d51.

Rebuild it with tools/extract_visual_anchors.py. Training-data metadata hashes
from the final experiment are stored in
../results/manifests/training_data.sha256.
