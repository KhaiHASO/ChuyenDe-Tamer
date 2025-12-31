import sys
import wandb
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI

from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER

def cli_main():
    # Handle WandB API Key from CLI (e.g., --wandb_api_key=XYZ or --wandb_api_key XYZ)
    # This allows passing the key without it being a valid LightningCLI argument
    if "--wandb_api_key" in sys.argv:
        try:
            idx = sys.argv.index("--wandb_api_key")
            key = sys.argv[idx + 1]
            wandb.login(key=key)
            print(f"Logged in to WandB with provided key.")
            # Remove arguments to prevent LightningCLI from crashing
            del sys.argv[idx]
            del sys.argv[idx] 
        except IndexError:
            print("Error: --wandb_api_key flag provided but no key found.")
    
    # Also support --wandb_api_key=XYZ format
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--wandb_api_key="):
            key = arg.split("=", 1)[1]
            wandb.login(key=key)
            print(f"Logged in to WandB with provided key.")
            del sys.argv[i]
            break

    cli = LightningCLI(
        LitTAMER,
        HMEDatamodule,
        save_config_overwrite=True,
        trainer_defaults={"plugins": DDPPlugin(find_unused_parameters=True)},
    )

if __name__ == "__main__":
    cli_main()