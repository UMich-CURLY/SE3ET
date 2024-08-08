CUDA_VISIBLE_DEVICES=0 python eval.py --benchmark=3DMatch --method=lgr

for n in 5000 2500 1000 500 250; do
    CUDA_VISIBLE_DEVICES=0 python eval.py --num_corr=$n --benchmark=3DMatch --method=ransac
done

CUDA_VISIBLE_DEVICES=0 python eval.py --benchmark=3DLoMatch --method=lgr

for n in 5000 2500 1000 500 250; do
    CUDA_VISIBLE_DEVICES=0 python eval.py --num_corr=$n --benchmark=3DLoMatch --method=ransac
done


