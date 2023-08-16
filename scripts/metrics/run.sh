
### Journal ###
root='results/'
out_root='results/metrics'

root='/group/30042/zhouxiawang/checkpoints/RestoreFormer/resutls/WebPhoto-Test'
out_root='/group/30042/zhouxiawang/checkpoints/RestoreFormer/metrics'

test_name='RestoreFormer++'

test_name='2022-11-16T14-55-40_RestoreFormer++RestoreFormer++_lmdb_gpus8_h4_E62_ROHQD105_seed32'
test_name='2022-11-21T10-49-14_RestoreFormer++RestoreFormer++_lmdb_gpus8_h4_E62_ROHQD105_En105_lr7_seed44'
tag='test'



test_image=$test_name'_'$tag'/restored_faces'
out_name=$test_name
need_post=1

CelebAHQ_GT='YOUR_PATH'


# FID
python -u scripts/metrics/cal_fid.py \
$root'/'$test_image \
--fid_stats 'experiments/pretrained_models/inception_FFHQ_512_color_gray008_shift02.pth' \
--save_name $out_root'/'$out_name'_fid_gray008shift02.txt' \

# --fid_stats 'experiments/pretrained_models/inception_FFHQ_512-f7b384ab.pth' \
# --save_name $out_root'/'$out_name'_fid.txt' \

# --fid_stats 'experiments/pretrained_models/inception_FFHQ_512_color_gray008_shift02.pth' \
# --save_name $out_root'/'$out_name'_fid_gray008shift02.txt' \

if [ -d $CelebAHQ_GT ]
then
    # PSRN SSIM LPIPS
    python -u scripts/metrics/cal_psnr_ssim.py \
    $root'/'$test_image \
    --gt_folder $CelebAHQ_GT \
    --save_name $out_root'/'$out_name'_psnr_ssim_lpips.txt' \
    --need_post $need_post \

    # # # PSRN SSIM LPIPS
    python -u scripts/metrics/cal_identity_distance.py  \
    $root'/'$test_image \
    --gt_folder $CelebAHQ_GT \
    --save_name $out_root'/'$out_name'_id.txt' \
    --need_post $need_post
else
    echo 'The path of GT does not exist'
fi