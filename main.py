#!/usr/bin/env python

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import logging
import json
import math
import os
import random
from pathlib import Path
import sys
from tqdm import tqdm
import wandb

import torch
from torch import nn
from torch.utils.data import DataLoader

import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed

from config import LLMConfig
from models import load_model_and_tokenizer
from dataset import make_dataset


logger = logging.getLogger(__name__)


"""TO-DO
dataset 토크나이징
dataloader 만들기
"""


@hydra.main(version_base=None, config_name="config")
def main(cfg: LLMConfig):
    
    # logging
    logger.info('Let\'s get started!!')
    
    accelerator_log_kwargs = {}
    if cfg.report.with_tracking:
        wandb.login(key=cfg.report.wandb_id, relogin=True)
        accelerator_log_kwargs["log_with"] = cfg.report.name
        accelerator = Accelerator(gradient_accumulation_steps=cfg.param.gradient_accumulation_steps, **accelerator_log_kwargs)
    else:
        accelerator = Accelerator(gradient_accumulation_steps=cfg.param.gradient_accumulation_steps)
    logger.info(accelerator.state)

    if cfg.param.seed is not None:
        set_seed(cfg.param.seed)

    # Model, tokenizer, Optimizer
    model, tokenizer = load_model_and_tokenizer(cfg.model.name)
    optimizer = torch.optim.AdamW(lr=cfg.param.lr)

    # # Create the data loaders
    train_dataset, valid_dataset = make_dataset(cfg.path.dataset_path)
    # train_loader = 
    # test_loader = 

    # # Run the epochs
    for epoch in cfg.param.epochs:
    


if __name__ == "__main__":
    main()