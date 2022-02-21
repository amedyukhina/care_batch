# command line
python care_batch/datagen.py --basepath /research/sharedresources/cbi/common/Anna/test_data/CARE/train --source_dir low --target_dir GT --save_file /home/amedyukh/data.npz --patch_size 32,64,64 --n_patches_per_image 20 --axes ZYX

python care_batch/train.py --data_file  /home/amedyukh/data.npz --model_basedir /research/sharedresources/cbi/common/Anna/test_data/CARE/model --model_name CARE --train_epochs 10 --train_steps_per_epoch 20

python care_batch/predict.py --model_basedir /research/sharedresources/cbi/common/Anna/test_data/CARE/model --model_name CARE --input_dir /research/sharedresources/cbi/common/Anna/test_data/CARE/test/low --output_dir /research/sharedresources/cbi/common/Anna/test_data/CARE/test/predicted --n_tiles 1,2,4


# on the cluster
bsub -P CARE -J generate_data -q standard -R "rusage[mem=10000]" "export PATH='/research/sharedresources/cbi/common/Anna/anaconda3/envs/csbdeep/bin/:$PATH'; python /home/amedyukh/codes/care_restoration/care_batch/generate_training_data.py --basepath /research/sharedresources/cbi/common/Anna/test_data/CARE/train --source_dir low --target_dir GT --save_file /home/amedyukh/data.npz --patch_size 32,64,64 --n_patches_per_image 20 --axes ZYX"

bsub -P CARE -J train -q gpu -R "rusage[ngpus_excl_p=1,mem=10G]" "export PATH='/research/sharedresources/cbi/common/Anna/anaconda3/envs/csbdeep/bin/:$PATH'; python /home/amedyukh/codes/care_restoration/care_batch/train.py --data_file  /home/amedyukh/data.npz --model_basedir /research/sharedresources/cbi/common/Anna/test_data/CARE/model --model_name CARE --train_epochs 10 --train_steps_per_epoch 20"

bsub -P CARE -J predict -q gpu -R "rusage[ngpus_excl_p=1,mem=10G]" "export PATH='/research/sharedresources/cbi/common/Anna/anaconda3/envs/csbdeep/bin/:$PATH'; python /home/amedyukh/codes/care_restoration/care_batch/predict.py --model_basedir /research/sharedresources/cbi/common/Anna/test_data/CARE/model --model_name CARE --input_dir /research/sharedresources/cbi/common/Anna/test_data/CARE/test/low --output_dir /research/sharedresources/cbi/common/Anna/test_data/CARE/test/predicted --n_tiles 1,2,4"

