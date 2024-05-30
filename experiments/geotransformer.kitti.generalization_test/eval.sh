# for n in 250 500 1000 2500 5000; do
#     CUDA_VISIBLE_DEVICES=0 python eval.py --num_corr=$n --benchmark=3DMatch --method=ransac
# done

CUDA_VISIBLE_DEVICES=0 python eval.py --benchmark=3DMatch --method=lgr

for n in 250 500 1000 2500 5000; do
    CUDA_VISIBLE_DEVICES=0 python eval.py --num_corr=$n --benchmark=3DLoMatch --method=ransac
done

CUDA_VISIBLE_DEVICES=0 python eval.py --benchmark=3DLoMatch --method=lgr

