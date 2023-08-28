## This is the official repo for ICCV Workshop paper "[MOFA: A Model Simplification Roadmap for Image Restoration on Mobile Devices](https://arxiv.org/abs/2308.12494)."
### Environment
```
pip install -r requirement.txt
python setup.py develop --no_cuda_ext
```
### Dataset 
Please check [NAFNet repo](https://github.com/megvii-research/NAFNet) for SIDD, GoPro and REDS, and [HINet repo](https://github.com/megvii-model/HINet) for Rain13k.
### To Train
```
sh run_script_train.sh
````
### To Test
```
sh run_script_test.sh
```
To test the runtime for PMRID and NAFNet baselines, we manually add split cat operations for fair comparison. An example can be found in `./basicsr/models/archs/PMRID_split_arch.py`.

### License
This project is under the MIT license, and it is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) which is under the Apache 2.0 license.

### Acknowledgements
Our codes are highly based on [NAFNet repo](https://github.com/megvii-research/NAFNet), [HINet repo](https://github.com/megvii-model/HINet) and [FUIR repo](https://github.com/murufeng/FUIR)
