# tinydiffusion
cs274E project repository

## DDIM
### To train ddim:
train.py args:

  --batch_size (default = 64)
  
  --epochs (default = 100)
  
  --lr (default = 1e-4)
  
  --noise_schedule ("linear" or "cosine", default = "linear")

  --model_path (What the saved model name will be called. default="model.pt")
```
cd ddim
python train.py --batch_size=32 --epochs=20 --model_path="mnist_model.pt"
```

### to sample:
ddim_sample.py args:

  --num_samples

  --inference_steps

  --noise_schedule

  --model_path

  --save_file

```
cd ddim
python sample_ddim.py --num_samples=10 --model_path="./results/ddim_mnist_model.pt" --save_file="images.png"
```
