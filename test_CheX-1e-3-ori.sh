CUDA_VISIBLE_DEVICES="0" \
python test_metrics_CheX_ori.py \
--checkpoint ./ckpts/CheX_1e-3/SAM2-UNet-20.pth \
--test_image_path ./data/CheXpert_SAM2UNet/CheXTestDataset/image/  \
--test_gt_path ./data/CheXpert_SAM2UNet/CheXTestDataset/masks/  \
--save_path ./results/CheX_1e-3_ori/