# Requierments
import logging as log
import yaml
import os
import sys
import torch
from pathlib import Path
import wandb
import fire
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor

# Dependencies
from src.setup import setup_folder_structure_FTYM
from src.dataset import ImageSegmentationDataset
from src.loss import PowerIoULoss
from src.fitter import SegmFitter
from src.callbacks import wandb_checkpoint
from src.utils import seed_everything

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


# Main method. Fire automatically allign method arguments with parse commands from console
def main(
        experiment_config:str="./config/experiment_config.yaml"
        ):

    #
    # Part I: Read configuration files
    #

    #Training
    with open(experiment_config) as file:
        config_dct = yaml.load(file, Loader=yaml.FullLoader)
        seed_everything(config_dct['train']['seed'])
    # Environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Create output path (if needed)
    Path(config_dct['train']['filepath']).mkdir(parents=True, exist_ok=True)

    #
    # Part II: Setup data and model
    #

    # Get tools
    log.debug(f"Setup tools:")
    train_idx, val_idx, tag2idx, class_weights = setup_folder_structure_FTYM()
    idx2tag = {v:k for k,v in tag2idx.items()}
    # Build datasets
    log.debug(f"Prepare datasets:")
    feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
    train_dts = ImageSegmentationDataset(os.path.join('./input', 'train'), train_idx, feature_extractor, config_dct['train']['img_size'], len(idx2tag), is_train=True)
    train_dtl = torch.utils.data.DataLoader(train_dts,
                                            batch_size=config_dct['train']['batch_size'],
                                            num_workers=config_dct['train']['num_workers'],
                                            shuffle=True,
                                            )
    val_dts = ImageSegmentationDataset(os.path.join('./input', 'val'), val_idx, feature_extractor, config_dct['train']['img_size'], len(idx2tag), is_train=False)
    val_dtl = torch.utils.data.DataLoader(val_dts,
                                          batch_size=2*config_dct['train']['batch_size'],
                                          num_workers=config_dct['train']['num_workers'],
                                          shuffle=False,
                                          )
    # Define model
    log.debug(f"Prepare model, loss function, optimizer and scheduler")
    model = SegformerForSemanticSegmentation.from_pretrained(
        config_dct['train']['hf_model'],
        ignore_mismatched_sizes=True,
        num_labels=len(tag2idx),
        id2label=idx2tag,
        label2id=tag2idx,
        reshape_last_stage=True
    )
    # Get loss, optimisers and schedulers
    loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights), ignore_index=config_dct['train']['ignore_index_loss'], reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config_dct['train']['learning_rate'],
                                  weight_decay=config_dct['train']['weight_decay'],
                                  )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=config_dct['train']['learning_rate'],
                                                    steps_per_epoch=len(train_dtl),
                                                    epochs=config_dct['train']['epochs'],
                                                    pct_start=config_dct['train']['warmup_epochs_factor'],
                                                    anneal_strategy='cos',
                                                    )
    # Fitter
    fitter = SegmFitter(
        model = model,
        loss = loss,
        optimizer = optimizer,
        scheduler = scheduler,
        validation_scheduler = False,
        step_scheduler = True,
        folder = config_dct['train']['filepath'],
        verbose = True,
        save_log = False,
        world_size = config_dct['train']['world_size'],
        mixed_precision = config_dct['train']['mixed_precision'],
        gradient_accumulation_steps = 1,
        clip_value = 1.
    )

    # Weights and Biases session
    wandb.login(key=config_dct['wandb']['WANDB_API_KEY'])
    wandb.init(project=config_dct['wandb']['WANDB_PROJECT'], entity=config_dct['wandb']['WANDB_USERNAME'], config=config_dct['train'])
    # Training
    log.debug(f"Start fitter training:")
    _ = fitter.fit(train_dtl = train_dtl,
                   val_dtl = val_dtl,
                   n_epochs = config_dct['train']['epochs'],
                   metrics = None,
                   early_stopping = config_dct['train']['early_stopping'],
                   early_stopping_mode = config_dct['train']['scheduler_mode'],
                   verbose_steps = config_dct['train']['verbose_steps'],
                   step_callbacks = [wandb_checkpoint],
                   validation_callbacks = [wandb_checkpoint]
                   )
    # Remove objects from memory
    del fitter, loss, optimizer, scheduler, train_dts, train_dtl

    #
    # Part V: Evaluation
    #

    # Move best checkpoint to Weights and Biases root directory to be saved
    log.debug(f"Move best checkpoint to Weights and Biases root directory to be saved:")
    os.replace(f"{config_dct['train']['filepath']}/best-checkpoint.bin", f"{wandb.run.dir}/best-checkpoint.bin")
    # End W&B session
    wandb.finish()
   

if __name__=="__main__":
    fire.Fire(main)