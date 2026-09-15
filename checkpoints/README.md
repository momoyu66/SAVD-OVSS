# External checkpoints

Model weights are intentionally not tracked. Place the following files in this
directory:

    eccv26_dinode_coco_stuff.pth
    open_clip_pytorch_model.bin
    facebook_dinov3_vitl16_pretrain_lvd1689m_hf/

For final h1536 paper reproduction, place the following externally distributed
student checkpoints here:

    flow_student_asfd_h1536_seed42.pth
    flow_student_asfd_h1536_seed123.pth
    flow_student_asfd_h1536_seed3407.pth
    flow_student_asfd_vadd_h1536_seed42.pth
    flow_student_asfd_vadd_h1536_seed123.pth
    flow_student_asfd_vadd_h1536_seed3407.pth

Their exact hashes are recorded in:

    ../results/manifests/runtime_weights.sha256

The original DINOv3 PyTorch checkpoint used to construct the local
Transformers directory was:

    dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth

Do not load untrusted PyTorch pickle checkpoints. Obtain weights from the
original authors or another source you trust, and verify their SHA-256 hashes.
