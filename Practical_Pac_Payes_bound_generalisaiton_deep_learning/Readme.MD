# Curvature

```
#SETUP
install iclr.yml environment
```


## Examples

```bash
#SGD
python3 run_sgd.py --dir=out/ --dataset=CIFAR100 --data_path=data/ --model=WideResNet28x10 --epochs=300 --lr_init=0.1 --wd=5e-4 --seed=1 --linear_annealing
#SGD with trace regularisation
python3 run_sgd_tracereg.py --dir=out/ --trace_factor 0.0001 --dataset=CIFAR100 --data_path=data/ --model=WideResNet28x10 --epochs=300 --lr_init=0.1 --wd=5e-4 --seed=1 --linear_annealing
```
