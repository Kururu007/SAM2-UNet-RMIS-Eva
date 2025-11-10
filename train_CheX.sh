CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path ./pretrained/sam2_hiera_large.pt \
--train_image_path ./data/CheXpert_SAM2UNet/CheXTrainDataset/image/ \
--train_mask_path ./data/CheXpert_SAM2UNet/CheXTrainDataset/masks/ \
--save_path ./ckpts/CheX_1e-3_20_512/ \
--epoch 20 \
--lr 0.001 \
--batch_size 8