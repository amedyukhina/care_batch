# command line
python scripts/datagen.py --basepath /research/sharedresources/cbi/common/Anna/test_data/CARE/train --source_dir low --target_dir GT --save_file /home/amedyukh/data.npz --patch_size 32,64,64 --n_patches_per_image 20 --axes ZYX

python scripts/train.py --data_file  /home/amedyukh/data.npz --model_basedir /research/sharedresources/cbi/common/Anna/test_data/CARE/model --model_name CARE --train_epochs 10 --train_steps_per_epoch 20

python scripts/restore.py --model_basedir /research/sharedresources/cbi/common/Anna/test_data/CARE/model --model_name CARE --input_dir /research/sharedresources/cbi/common/Anna/test_data/CARE/test/low --output_dir /research/sharedresources/cbi/common/Anna/test_data/CARE/test/predicted --n_tiles 1,2,4

