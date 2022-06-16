
python main_pretext.py '/ssd/imagenet/ILSVRC/Data/CLS-LOC' --dim 128 --dataset mri_brain -a resnet18 --cos --warm --lincls --tb --resume true --method npair  --imix icutmix  --proj mlp --temp 0.02 --epochs 800 --trial 14 --multiprocessing-distributed --dist-url 'tcp://localhost:10033' --lr 0.003 -b 128 --qlen 65536 --class-ratio 0.1
