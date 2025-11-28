# tinydiffusion
cs274E project repository

To run ddim:
train.py args:
  --batch_size (default = 64)
  --epochs (default = 100)
  --lr (default = 1e-4)
  --noise_schedule ("linear" or "cosine", default = "linear")
```
cd ddim
python train.py --batch_size=32 --epochs=20
```
