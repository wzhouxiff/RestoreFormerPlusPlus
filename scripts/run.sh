export BASICSR_JIT=True

# For RestoreFormer
# conf_name='HQ_Dictionary'
# conf_name='RestoreFormer'

# For RestoreFormer++
conf_name='ROHQD'
conf_name='RestoreFormerPlusPlus'

# gpus='0,1,2,3,4,5,6,7'
# node_n=1
# ntasks_per_node=8

root_path='PATH_TO_CHECKPOINTS'

gpus='0,'
node_n=1
ntasks_per_node=1

gpu_n=$(expr $node_n \* $ntasks_per_node)

python -u main.py \
--root-path $root_path \
--base 'configs/'$conf_name'.yaml' \
-t True \
--postfix $conf_name'_gpus'$gpu_n \
--gpus $gpus \
--num-nodes $node_n \
--random-seed True \
