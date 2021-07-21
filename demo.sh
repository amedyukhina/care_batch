python care_batch/generate_training_data.py --basepath /research/sharedresources/cbi/common/Anna/test_data/CARE/train --source_dir low --target_dir GT --save_file /home/amedyukh/data.npz --patch_size 32,64,64 --n_patches_per_image 20 --axes ZYX

python care_batch/train.py --data_file  /home/amedyukh/data.npz --model_basedir /research/sharedresources/cbi/common/Anna/test_data/CARE/model --model_name CARE --train_epochs 10 --train_steps_per_epoch 20
