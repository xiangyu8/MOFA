python -m torch.distributed.run --nproc_per_node=8  basicsr/train.py -opt options/train/SIDD/PMRID.yml --launcher pytorch
