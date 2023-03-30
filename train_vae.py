from pythae.pipelines import TrainingPipeline
from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig

from models import TftEncoder, TftDecoder
from tft_vae import TftVAE
from tft_dataset import TftDataset

# Set up the training configuration
my_training_config = BaseTrainerConfig(
    output_dir='my_model',
    num_epochs=50,
    learning_rate=1e-3,
    per_device_train_batch_size=200,
    per_device_eval_batch_size=200,
    train_dataloader_num_workers=0,
    eval_dataloader_num_workers=0,
    steps_saving=20,
    optimizer_cls="AdamW",
    optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.995)},
    scheduler_cls="ReduceLROnPlateau",
    scheduler_params={"patience": 5, "factor": 0.5}
)
# Set up the model configuration 
my_vae_config = model_config = VAEConfig(
    input_dim=(1, 28, 28),
    latent_dim=10
)
# Build the model
my_vae_model = TftVAE(
    model_config=my_vae_config,
    encoder = TftEncoder(),
    decoder = TftDecoder()
)
# Build the Pipeline
pipeline = TrainingPipeline(
	training_config=my_training_config,
	model=my_vae_model
)
# Launch the Pipeline
pipeline(
    train_data=TftDataset(), # must be torch.Tensor, np.array or torch datasets
    eval_data=TftDataset() # must be torch.Tensor, np.array or torch datasets
)