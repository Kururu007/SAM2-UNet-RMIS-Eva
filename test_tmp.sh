CUDA_VISIBLE_DEVICES="0" \
python test_metrics.py \
--checkpoint ./ckpts/tmp/SAM2-UNet-20.pth \
--test_image_path ./data/TestDataset/Kvasir/images/  \
--test_gt_path ./data/TestDataset/Kvasir/masks/  \
--save_path ./results/tmp/Kvasir/