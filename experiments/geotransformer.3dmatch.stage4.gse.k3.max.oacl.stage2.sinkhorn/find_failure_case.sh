for n in 600 890 940 950 960;
do
    CUDA_VISIBLE_DEVICES=0 python demo.py --src_file=../../data/demo/src_$n.npy --ref_file=../../data/demo/ref_$n.npy --gt_file=../../data/demo/gt_$n.npy --weights=../../weights/geotransformer-3dmatch.pth.tar
done

