CUDA_VISIBLE_DEVICES="0" \
python test_metrics_QaTa.py \
--checkpoint ./ckpts/QaTa/SAM2-UNet-20.pth \
--test_image_path ./data/QaTa_SAM2UNet/QaTaTestDataset/image/  \
--test_gt_path ./data/QaTa_SAM2UNet/QaTaTestDataset/masks/  \
--save_path ./results/QaTa/