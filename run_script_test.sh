python -m torch.distributed.run --nproc_per_node=8  basicsr/test.py -opt options/test/SIDD/NAFNet-width16_mofa.yml  --launcher pytorch
python -m torch.distributed.run --nproc_per_node=8  basicsr/test.py -opt options/test/GoPro/NAFNet-width16_mofa.yml  --launcher pytorch
python -m torch.distributed.run --nproc_per_node=8  basicsr/test.py -opt options/test/REDS/NAFNet-width16_mofa.yml  --launcher pytorch
