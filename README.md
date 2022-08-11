# Pixel to Binary Embedding TowardsRobustness for CNNs
The repository is the official implementation of [Pixel to Binary Embedding](https://arxiv.org/abs/2206.05898).

![overview-1](https://user-images.githubusercontent.com/10476902/184076571-5e88424d-b8c7-4c23-9a48-af0aa3be0ee7.png)



## Requirements

Please download CIFAR-100-C by the url below:
```
    CIFAR-100-C: https://zenodo.org/record/3555552#.YLX3THUzakA
```

## Training

#### Common Visual Corruptions on CIFAR-100-C

###### P2BE(M=64, λ=1.0) + WideResNet/ DenseNet-BC (k=12,d=100)/ ResNeXt-29 (32×4) + Augmix
```
python cifar_augmix.py --corrupted path_to_cifar100-c --dataset cifar100 -m wrnb -e 200 -s results_wrnb_cifar100 --m 64 --coefficient_smooth 1.0 --wde 1.0e-4 --lre 1.0e-4
python cifar_augmix.py --corrupted path_to_cifar100-c --dataset cifar100 -m densenetb -e 200 -s results_densenetb_cifar100 --m 64 --coefficient_smooth 1.0 --wde 1.0e-4 --lre 1.0e-4
python cifar_augmix.py --corrupted path_to_cifar100-c --dataset cifar100 -m resnextb -e 200 -s results_densenetb_cifar100 --m 64 --coefficient_smooth 1.0 --wde 1.0e-4 --lre 1.0e-4
```

###### One-hot(M=64) + WideResNet / DenseNet-BC (k=12,d=100) / ResNeXt-29 (32×4) + Augmix
```
python cifar_augmix.py --dataset cifar100 --corrupted path_to_cifar100-c -m wrnonehot -e 200 -s cifar100_wrn_onehot --m 64
python cifar_augmix.py --dataset cifar100 --corrupted path_to_cifar100-c -m densenetonehot -e 200 -s cifar100_densenet_onehot --m 64
python cifar_augmix.py --dataset cifar100 --corrupted path_to_cifar100-c -m resnextonehot -e 200 -s cifar100_resnext_onehot --m 64
```

###### Thermometer(M=64) + WideResNe t/ DenseNet-BC (k=12,d=100) / ResNeXt-29 (32×4) + Augmix
```
python cifar_augmix.py --dataset cifar100 --corrupted path_to_cifar100-c -m wrnthermo -e 200 -s cifar100_wrn_thermo --m 64
python cifar_augmix.py --dataset cifar100 --corrupted path_to_cifar100-c -m densenetthermo -e 200 -s cifar100_densenet_thermo --m 64
python cifar_augmix.py --dataset cifar100 --corrupted path_to_cifar100-c -m resnextthermo -e 200 -s cifar100_resnext_thermo --m 64
```


###### RGB + WideResNe t/ DenseNet-BC (k=12,d=100) / ResNeXt-29 (32×4) + Augmix
```
python cifar_augmix.py --dataset cifar100 --corrupted path_to_cifar100-c -m wrn -e 200 -s cifar100_wrn_rgb --m 64
python cifar_augmix.py --dataset cifar100 --corrupted path_to_cifar100-c -m densenet -e 200 -s cifar100_densenet_rgb --m 64
python cifar_augmix.py --dataset cifar100 --corrupted path_to_cifar100-c -m resnext -e 200 -s cifar100_resnext_rgb --m 64
```

#### Adversarial Perturbations on CIFAR-datasets

######  P2BE(M=64, λ=1.0) / One-hot(M=64) / Thermometer(M=64) + ConTrain + CIFAR-10

```
python cifar_advtrain.py --dataset cifar10 -m wrnb -e 200 -s cifar10_contrain_wrn_p2be --m 64 --coefficient_smooth 1.0 --wde 1.0e-4 --lre 1.0e-4 --batch-size 128 --split 128 --xi 1.0
python cifar_advtrain.py --dataset cifar10 -m wrnonehot -e 200 -s cifar10_contrain_wrnonehot --m 64 --batch-size 128 --split 128 --xi 1.0
python cifar_advtrain.py --dataset cifar10 -m wrnthermo -e 200 -s cifar10_contrain_wrnthermo --m 64 --batch-size 128 --split 128 --xi 1.0
```

######  P2BE(M=64, λ=0.1) / One-hot(M=64) / Thermometer(M=64) + ConTrain + CIFAR-100

```
python cifar_advtrain.py --dataset cifar100 -m wrnb -e 200 -s cifar100_contrain_wrn_p2be --m 64 --coefficient_smooth 0.1 --wde 1.0e-4 --lre 1.0e-4 --batch-size 128 --split 128 --xi 1.0
python cifar_advtrain.py --dataset cifar100 -m wrnonehot -e 200 -s cifar100_contrain_wrnonehot --m 64 --batch-size 128 --split 128 --xi 1.0
python cifar_advtrain.py --dataset cifar100 -m wrnthermo -e 200 -s cifar100_contrain_wrnthermo --m 64 --batch-size 128 --split 128 --xi 1.0
```
######  P2BE(M=64, λ=1.0) / One-hot(M=64) / Thermometer(M=64) + AdvTrain + CIFAR-10

```
python cifar_advtrain.py --dataset cifar10 -m wrnb -e 200 -s cifar10_advtrain_wrn_p2be --m 64 --coefficient_smooth 1.0 --wde 1.0e-4 --lre 1.0e-4 --batch-size 128 --split 128 --xi 1.0 --mode onlyadv
python cifar_advtrain.py --dataset cifar10 -m wrnonehot -e 200 -s cifar10_advtrain_wrnonehot --m 64 --batch-size 128 --split 128 --xi 1.0 --mode onlyadv
python cifar_advtrain.py --dataset cifar10 -m wrnthermo -e 200 -s cifar10_advtrain_wrnthermo --m 64 --batch-size 128 --split 128 --xi 1.0 --mode onlyadv
```

######  P2BE(M=64, λ=0.1) / One-hot(M=64) / Thermometer(M=64) + AdvTrain + CIFAR-100

```
python cifar_advtrain.py --dataset cifar100 -m wrnb -e 200 -s cifar100_advtrain_wrn_p2be --m 64 --coefficient_smooth 0.1 --wde 1.0e-4 --lre 1.0e-4 --batch-size 128 --split 128 --xi 1.0 --mode onlyadv
python cifar_advtrain.py --dataset cifar100 -m wrnonehot -e 200 -s cifar100_advtrain_wrnonehot --m 64 --batch-size 128 --split 128 --xi 1.0 --mode onlyadv
python cifar_advtrain.py --dataset cifar100 -m wrnthermo -e 200 -s cifar100_advtrain_wrnthermo --m 64 --batch-size 128 --split 128 --xi 1.0 --mode onlyadv
```


## Citation
If you find this useful for your work, please consider citing
```

@article{p2be,
  title = {Pixel to Binary Embedding Towards Robustness for CNNs},
  author = {Kishida, Ikki and Nakayama, Hideki},
  journal={ICPR},
  year={2022}
}

```
