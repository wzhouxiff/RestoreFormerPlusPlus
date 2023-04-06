export BASICSR_JIT=True

conf_name='HQ_Dictionary'
# conf_name='RestoreFormer'
conf_name='ROHQD'
conf_name='RestoreFormer++'

gpus='0,1,2,3,4,5,6'
# gpus='0,1,2,3,4,5,6,7'
node_n=1
ntasks_per_node=7

gpus='7,'
node_n=1
ntasks_per_node=1

gpu_n=$(expr $node_n \* $ntasks_per_node)

python -u main.py \
--root-path /group/30042/zhouxiawang/checkpoints/RestoreFormer/release \
--base 'configs/'$conf_name'.yaml' \
-t True \
--postfix $conf_name'_gpus'$gpu_n'_lmdb_h4_ROHQD105' \
--gpus $gpus \
--num-nodes $node_n \
--random-seed True \
