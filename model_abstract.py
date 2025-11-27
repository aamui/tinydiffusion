class Model:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("This is an abstract base class.")
    
    def train(self, X_train, y_train, X_test, y_test, num_epochs=1, use_wandb=True, 
              device='cpu', batch_size=32):
        raise NotImplementedError("Train method must be implemented by subclasses.")
    
    def generate(self, num_samples, device='cpu', number_of_steps=25):
        raise NotImplementedError("Generate method must be implemented by subclasses.")
    
    def generate_dataset(self, num_samples, number_of_steps=100, device='cpu', 
                         max_images_per_batch=2048, sample_shape=(28, 28)):
        raise NotImplementedError("Generate dataset method must be implemented by subclasses.")

