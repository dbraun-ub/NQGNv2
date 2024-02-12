CUDA_VISIBLE_DEVICES=1 python ../train.py --model_name resnet18-QuadtreeSpConv2-0.004-20-epochs \
--data_path=/media/HDD1/datasets/Kitty \
--log_dir /media/HDD3/dbraun/train/NQGNv2 \
--stereoNet_path /media/HDD1/train_daniel/stereoNet/stereoNet_MS_640x192_b10/models/weights_29 \
--load_weights_EPCDepth_encoder /media/HDD1/train_daniel/EPCDepth/model18_192x640.pth.tar \
--arch_encoder resnet18 --arch_decoder QuadtreeSpConv2 \
--use_stereo --frame_ids 0 --batch_size 10 --num_workers 3 \
--coef_quadtree 0.2 --coef_l1 0.8 --coef_rep 0 --disparity_smoothness 0 \
--scales 0 1 2 3 4 5 --crit 0.004 --num_epochs 20 --use_labels
