CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path ./pretrained/sam2_hiera_large.pt \
--train_image_path ./data/TrainDataset/image/ \
--train_mask_path ./data/TrainDataset/masks/ \
--save_path ./ckpts/tmp/ \
--epoch 20 \
--lr 0.001 \
--batch_size 16