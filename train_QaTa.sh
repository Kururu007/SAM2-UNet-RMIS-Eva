CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path ./pretrained/sam2_hiera_large.pt \
--train_image_path ./data/QaTa_SAM2UNet/QaTaTrainDataset/image/ \
--train_mask_path ./data/QaTa_SAM2UNet/QaTaTrainDataset/masks/ \
--save_path ./ckpts/QaTa/ \
--epoch 20 \
--lr 0.0001 \
--batch_size 16