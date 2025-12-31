import os
import wandb
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger

from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER

# Kaggle / WandB Setup
# Try to get key from Kaggle Secrets if available, otherwise check env var or hardcoded backup
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    WANDB_API_KEY = user_secrets.get_secret("wandb_api_key")
    print("Found WandB API key in Kaggle Secrets.")
except ImportError:
    pass
except Exception as e:
    print(f"Could not load key from Kaggle Secrets: {e}")

# FALLBACK (User Provided Key) - intended for immediate use, safe to remove if using secrets
if not WANDB_API_KEY:
    # TODO: Remove this hardcoded key before sharing publicly if preferred.
    WANDB_API_KEY = "3010beaefcbb3ca747099418f4dd36cd474cc81c"
    print("Using hardcoded WandB API Key.")

if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)


def cli_main():
    # Chỉ log model khi nó lọt top tốt nhất (dựa trên config checkpoint)
    wandb_logger = WandbLogger(project="TAMER-Kaggle", log_model=True)
    
    cli = LightningCLI(
        LitTAMER,
        HMEDatamodule,
        save_config_overwrite=True,
        trainer_defaults={
            "plugins": DDPPlugin(find_unused_parameters=True),
            "logger": wandb_logger,
            "log_every_n_steps": 50, # Log more frequently
        },
    )

if __name__ == "__main__":
    cli_main()
