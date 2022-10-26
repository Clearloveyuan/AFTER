#datadir, labelpath, savingpath, GPUid, outputname, numexps
if [ -d "$3" ]; then
    echo "$3 directory already exists!";
    exit 1;
fi
CUDA_VISIBLE_DEVICES=$4 python run_baseline_continueFT.py --saving_path $3 --precision 16 --datadir $1 --labelpath $2 --training_step 20000  --warmup_step 100 --save_top_k 1 --lr 1e-4 --batch_size 64 --use_bucket_sampler 
