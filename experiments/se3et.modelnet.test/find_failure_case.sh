for n in 995 994 960 950 940 890 600;
do
    CUDA_VISIBLE_DEVICES=0 python demo.py --src_file=../../data/demo/src_$n.npy --ref_file=../../data/demo/ref_$n.npy --gt_file=../../data/demo/gt_$n.npy --weights=../../weights/e2pntransformer.3dmatch.exp53.pth.tar
done

