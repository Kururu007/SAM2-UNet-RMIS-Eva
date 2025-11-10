CUDA_VISIBLE_DEVICES="0" \
python test_metrics_QaTa_224.py \
--checkpoint ./ckpts/QaTa_1e-4/SAM2-UNet-20.pth \
--test_image_path ./data/QaTa_SAM2UNet/QaTaTestDataset/image/  \
--test_gt_path ./data/QaTa_SAM2UNet/QaTaTestDataset/masks/  \
--save_path ./results/QaTa_1e-4_224/