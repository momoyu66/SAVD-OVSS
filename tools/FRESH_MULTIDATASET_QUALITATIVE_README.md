# Fresh multi-dataset qualitative comparison

This kit creates new qualitative examples directly from the validation sets.
It does not reuse screenshots or masks from previous papers.

The candidate pool contains six images from each of PASCAL VOC 2012,
PASCAL Context, COCO-Stuff, and ADE20K. Candidate ranking uses ground-truth
annotation complexity only; predictions are not inspected during this stage.

Every candidate is then evaluated by:

- ClearCLIP
- NACLIP
- ProxyCLIP
- GLA-CLIP
- ASFD+VADD (h1536, seed 123)

ClearCLIP, NACLIP, ProxyCLIP, and GLA-CLIP run through the same official
GLA-CLIP framework and dataset pipeline. Raw label maps are exported so the
final paper figure can apply one palette and one opacity to all methods.

## Run on the server

```bash
cd ~/qx/DINOde
tar -xzf fresh_multidataset_qualitative_server_kit.tar.gz
chmod +x tools/run_fresh_multidataset_qualitative.sh
bash tools/run_fresh_multidataset_qualitative.sh
```

The first run may download the official OpenAI CLIP ViT-B/16 and DINO ViT-B/8
weights if they are not already in the PyTorch cache. The script checks Python
dependencies before starting inference and reports any missing package.

After all runs finish, download:

```text
outputs/fresh_multidataset_qualitative_results.tar.gz
```

The archive includes the 24 fresh RGB images, ground-truth masks, all raw
predictions, the prediction-blind selection metadata, and per-image diagnostic
scores used only to review candidates for the final four-row figure.
