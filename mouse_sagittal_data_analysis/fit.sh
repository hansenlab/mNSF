
module avail python
module load python/3.9.10
module load cudnn
nvidia-smi

export CUDA_VISIBLE_DEVICES=3
TF_GPU_ALLOCATOR=cuda_malloc_async


python3 fit.py

wait

printf "finish training"




