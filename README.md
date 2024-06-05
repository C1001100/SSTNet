# SSTNet for remote sensing image super-resolution
Official Pytorch implementation of the paper "[Activated Sparsely Sub-Pixel Transformer for Remote Sensing Image Super-Resolution](https://www.mdpi.com/2072-4292/16/11/1895)" accepted by Remote Sensing.  


## Requirements
- Python 3.6+
- Pytorch>=1.6
- torchvision>=0.7.0
- einops
- matplotlib
- cv2
- scipy
- tqdm
- scikit


## Installation
Clone or download this code and install aforementioned requirements 
```
cd codes
```

## Train
Download the UCMerced dataset[[Baidu Drive](https://pan.baidu.com/s/1ijFUcLozP2wiHg14VBFYWw),password:terr][[Google Drive](https://drive.google.com/file/d/12pmtffUEAhbEAIn_pit8FxwcdNk4Bgjg/view)]and AID dataset[[Baidu Drive](https://pan.baidu.com/s/1Cf-J_YdcCB2avPEUZNBoCA),password:id1n][[Google Drive](https://drive.google.com/file/d/1d_Wq_U8DW-dOC3etvF4bbbWMOEqtZwF7/view)], they have been split them into train/val/test data, where the original images would be taken as the HR references and the corresponding LR images are generated by bicubic down-sample. 
```
# x4
python demo_train.py --model=Proposed --dataset=UCMerced --scale=4 --patch_size=192 --ext=img --save=TRANSENETx4_UCMerced
# x3
python demo_train.py --model=Proposed --dataset=UCMerced --scale=3 --patch_size=144 --ext=img --save=TRANSENETx3_UCMerced
# x2
python demo_train.py --model=Proposed --dataset=UCMerced --scale=2 --patch_size=96 --ext=img --save=TRANSENETx2_UCMerced

python demo_train.py --model=Proposed --dataset=AID --scale=2 --patch_size=96 --ext=img --save=TRANSENETAHH
```

The train/val data pathes are set in [data/__init__.py](codes/data/__init__.py) 

## Test 
The test data path and the save path can be edited in [demo_deploy.py](codes/demo_deploy.py)

```
# x4
python demo_deploy.py --model=Proposed --scale=4
# x3
python demo_deploy.py --model=Proposed --scale=3
# x2
python demo_deploy.py --model=Proposed --scale=2

```

## Evaluation 
Compute the evaluated results in term of PSNR and SSIM, where the SR/HR paths can be edited in [calculate_PSNR_SSIM.py](codes/metric_scripts/calculate_PSNR_SSIM.py)

```
cd metric_scripts 
python calculate_PSNR_SSIM.py
```

## Citation 
If you find this code useful for your research, please cite our paper:
``````
@article{guo2024activated,
  title={Activated Sparsely Sub-Pixel Transformer for Remote Sensing Image Super-Resolution},
  author={Guo, Yongde and Gong, Chengying and Yan, Jun},
  journal={Remote Sensing},
  volume={16},
  number={11},
  pages={1895},
  year={2024},
  publisher={MDPI}
}
``````

## Acknowledgements 
This code is built on [TransENet (Pytorch)](https://github.com/Shaosifan/TransENet), [RCAN (Pytorch)](https://github.com/yulunzhang/RCAN) and [EDSR (Pytorch)](https://github.com/sanghyun-son/EDSR-PyTorch). We thank the authors for sharing the codes.  

