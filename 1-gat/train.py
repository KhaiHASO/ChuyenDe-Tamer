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
    # 1. Logger
    wandb_logger = WandbLogger(project="TAMER-Kaggle", log_model=True)
    
    # 2. Checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath="/kaggle/working/lightning",
        filename="crohme-{epoch:02d}-{val_ExpRate:.4f}",
        save_top_k=3,
        monitor="val_ExpRate",
        mode="max",
        save_last=True
    )

    cli = LightningCLI(
        LitTAMER,
        HMEDatamodule,
        save_config_callback=None, # <--- QUAN TRỌNG: Thêm dòng này để tắt lỗi AssertionError
        save_config_overwrite=True,
        trainer_defaults={
            "plugins": DDPPlugin(find_unused_parameters=True),
            "logger": wandb_logger,
            "callbacks": [checkpoint_callback],
            "log_every_n_steps": 20,
            # Thêm dòng này để ép chạy DDP chuẩn (tránh warning ddp_spawn)
            "strategy": "ddp", 
        },
    )
if __name__ == "__main__":
    cli_main()
