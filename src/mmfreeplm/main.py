import os
import yaml
import time
import argparse
from mmfreelm.models import HGRNBitConfig, HGRNBitModel
from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitPreTrainedModel
from mmfreelm.modules.fused_bitlinear import FusedBitLinear
import torch
from lightning.fabric import Fabric
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
torch.set_float32_matmul_precision('medium')

from datasets import ProteinDataset, PadBatch, TokenBatchSampler, TOKENS,TokenBatchSamplerStL,ProteinMemMapDataset,MemMapBatchSampler,ProteinDatasetESM1,TOKENS_ESM1,ProteinDatasetNLP
from training import train_model

from lightning.fabric.wrappers import _FabricOptimizer
from torch.optim import Optimizer
from typing import Any, Callable, Dict, Optional, List

#class CustomFabricOptimizer(_FabricOptimizer):
#    def __init__(self, optimizer: Optimizer, strategy, callbacks: Optional[List[Any]] = None):
#        super().__init__(optimizer, strategy, callbacks)
#        self._refresh()

#    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Load the state into the optimizer
#        self.optimizer.load_state_dict(state_dict)
        # Refresh the wrapper's internal state
#        self._refresh()

#    def _refresh(self) -> None:
#        """Refresh the wrapper's internal state to match the optimizer's state."""
#        self.__dict__.update(
#            {
#                k: v
#                for k, v in self.optimizer.__dict__.items()
#                if k not in ("load_state_dict", "state_dict", "step", "__del__")
#            }
#        )

class ChannelNorm (torch.nn.Module):
    def __init__(self, c_max=1,c_min=-1):
        super().__init__()
        self.c_max = c_max
        self.c_min = c_min
        self.scale = (c_max - c_min)
        
    def forward(self,logits):
        x_max,_ = torch.max(logits,dim=-1,keepdim=True)
        x_min,_ = torch.min(logits,dim=-1,keepdim=True)
        logits_norm = (logits - x_min) / (x_max - x_min)
        logits_scaled = logits_norm * self.scale + self.c_min
        return logits_scaled

def smooth_quant(x,weight,alpha=0.5):
    #Matmul Version, Temp
    max_act = torch.max(x, dim=1)[0].pow(alpha)
    max_weight = torch.max(weight, dim=0, keepdim=True)[0].pow(1-alpha)
    scale_fact = max_act/max_weight
    diag_s = torch.diag(scale_fact)
    inv_diag_s = torch.pow(diag_s,-1)
    x_hat = x @ inv_diag_s
    weight_hat = diag_s @ weight
    return x_hat,weight_hat


class OutlierSmoothing (torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        
    def forward(self,logits):

        weight_linear = self.linear.weight
        logits_hat, weight_hat = smooth_quant(logits,weight_linear)
        self.linear.weight = weight_hat
        logits_coma = self.linear(logits_hat)
        return logits_coma
    

class HGRNBitForMLM(HGRNBitPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, extension):
        super().__init__(config)
        self.model = HGRNBitModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = FusedBitLinear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        if extension == "progressive_unfreezing":
            extra_layer = torch.nn.TransformerEncoderLayer(d_model=config.hidden_size,nhead=config.num_heads, bias=False,
                                                            dim_feedforward=config.hidden_size*4,batch_first=True,norm_first=True)
            self.lm_head = torch.nn.Sequential(*[extra_layer,self.lm_head])
            self.model.requires_grad = False
        elif extension == "adapter_pretraining":
            extra_layer = torch.nn.TransformerEncoderLayer(d_model=config.hidden_size,nhead=config.num_heads, bias=False,
                                                            dim_feedforward=config.hidden_size*4,batch_first=True,norm_first=True)
            norm = ChannelNorm()
            self.lm_head = torch.nn.Sequential(*[extra_layer,norm,self.lm_head])


        elif extension == "outlier_smoothing":
            lm_layer = OutlierSmoothing(FusedBitLinear(config.hidden_size, config.vocab_size, bias=False))
            self.lm_head = lm_layer
        else:
            pass
    def forward(self,x):
        outputs = self.model(x)
        hidden_states = outputs[0]
        #logits = self.lm_head(hidden_states, flag = 0)
        logits = self.lm_head(hidden_states)

        return logits

def main(config, from_checkpoint, extra_epoch):
    # Initialize Wandb and Fabric modules
    wandb_logger = WandbLogger(
        project=config['wandb_project'],
        name=config['wandb_name'],
        # id is the name and a random number
        id=config['wandb_name'] + "-" + str(torch.randint(0, 1000000, (1,)).item()),
        config=config)
    
    fabric = Fabric(
        accelerator="gpu",
        num_nodes=config["num_nodes"],
        devices=4,
        precision=config['precision'],
        strategy="deepspeed",
        loggers=wandb_logger)

    # Launch Fabric
    fabric.launch()
    if config["esm1_like"] is True:
        tokens = TOKENS_ESM1
    else:
        tokens = TOKENS
    # Model and optimizer setup
    '''model_config = HGRNBitConfig(
        vocab_size=len(tokens),
        hidden_size=config['embed_dim'],
        num_hidden_layers=config['num_layers'],
        num_heads=config['num_heads'],
        pad_token_id=tokens.index('+'),
        bos_token_id=tokens.index('<'),
        eos_token_id=tokens.index('>'),
        attn_mode = config['attn_mode']
    )'''

    model_config = HGRNBitConfig(
        vocab_size=len(tokens),
        hidden_size=config['embed_dim'],
        num_hidden_layers=config['num_layers'],
        num_heads=config['num_heads'],
        pad_token_id=tokens.index('+'),
        bos_token_id=tokens.index('<'),
        eos_token_id=tokens.index('>'),
        attn_mode = config['attn_mode']
    )

    def _fabric_optimizer_load_state_dict(self, state_dict):
    # Load the state into the underlying optimizer
        self.optimizer.load_state_dict(state_dict)
    # Refresh the wrapper's internal state
        self.__dict__.update(
            {
                k: v
                for k, v in self.optimizer.__dict__.items()
                if k not in ("load_state_dict", "state_dict", "step", "__del__")
            }
        )

    def _fabric_optimizer_refresh(self):
        """Refreshes the wrapper's internal state to match the optimizer's state.
        This is needed to keep the wrapper in sync after loading the state dictionary.
        """

        self.__dict__.update(
            {
                k: v
                for k, v in self.optimizer.__dict__.items()
                if k not in ("load_state_dict", "state_dict", "step", "__del__")
            }
        )


    
    _FabricOptimizer.load_state_dict = _fabric_optimizer_load_state_dict
    _FabricOptimizer._refresh = _fabric_optimizer_refresh

    model = HGRNBitForMLM(model_config, config["extension_type"])
    fabric.print(model.config)
    fabric.print(model)
    fabric.print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    initial_lr = config['initial_lr']
    wd = config['weight_decay']
    fabric.print(f"Using initial learning rate: {initial_lr}, weight decay: {wd}")
    # optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, betas=(0.9, 0.98), weight_decay=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, betas=(0.9, 0.95), weight_decay = wd, eps=1e-15)
    # optimizer = CustomFabricOptimizer(optimizer, fabric._strategy)
    model, optimizer = fabric.setup(model, optimizer)
    
    #optimizer = CustomFabricOptimizer(optimizer.optimizer, fabric._strategy)
    
    # Load model weights if specified
    state = {"model": model, "optimizer": optimizer, "step": 0, "best_loss": float('inf')}

    if from_checkpoint or extra_epoch:
        checkpoint_path = config['checkpoint']
        fabric.load(checkpoint_path, state)
        # full_checkpoint = fabric.load(checkpoint_path)
        
        #model.load_state_dict(full_checkpoint["model"])
        #optimizer.load_state_dict(full_checkpoint["optimizer"])

        # After loading the checkpoint
        optimizer_state_dict = optimizer.state_dict()
        if not optimizer_state_dict.get('state') and not optimizer_state_dict.get('base_optimizer_state'):
            fabric.print("Optimizer state is empty after loading the checkpoint.")
        else:
            fabric.print("Optimizer state loaded successfully.")

        #  print(optimizer_state_dict)
        #  fabric.print("Optimizer state dict keys:", optimizer_state_dict.keys())

        fabric.print("Loaded weights from:", config['checkpoint'])
        fabric.print(optimizer.state_dict())
        if extra_epoch:
            state["step"] = 0

        if 'base_optimizer_state' in optimizer_state_dict:
            base_optimizer_state = optimizer_state_dict['base_optimizer_state']
            if base_optimizer_state['state']:
                fabric.print("Optimizer state loaded successfully.")
            else:
                fabric.print("Optimizer state is empty after loading the checkpoint.")
        else:
            fabric.print("Optimizer state loaded successfully.")

    #print(optimizer.state_dict())

    # Check if the state dictionary is non-empty
    # if not optimizer_state_dict['state']:
    #     fabric.print("Optimizer state is empty after loading the checkpoint.")
    # else:
    #     fabric.print("Optimizer state loaded successfully.")

    fabric.print("Checking step number:", state["step"], "and best loss:", state["best_loss"])

    #model, optimizer = fabric.setup(model, optimizer)
    # Loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=model.model.padding_idx)

    # Prepare train dataset
    fabric.print("Loading dataset...")
    since = time.time()
    
    if config["esm1_like"] is True:
        if config["nlp_mode"] is True:
            train_dataset = ProteinDatasetNLP(config['train_data'])
        else:
            train_dataset = ProteinDatasetESM1(config['train_data'])
    else:
        train_dataset = ProteinDataset(config['train_data'])
    
    fabric.print(f"Read file with {len(train_dataset)} sequences.")
    start_batch = state["step"] * config["num_nodes"] * 4 # 4 is the number of GPUs per node
    
    if config["esm1_like"] is True:
        train_sampler = TokenBatchSamplerStL(train_dataset, start_batch=start_batch, max_tokens=config['toks_per_batch'])
    else:
        train_sampler = TokenBatchSamplerStL(train_dataset, start_batch=start_batch, max_tokens=config['toks_per_batch'])
    
    train_data_loader = DataLoader(train_dataset, sampler=train_sampler, collate_fn=PadBatch, drop_last=True, pin_memory=True)
    fabric.print(f"Sampler prepared with {len(train_sampler)} batches (starting from batch {start_batch}).")
    train_data_loader = fabric.setup_dataloaders(train_data_loader)
    data_loaders = {"train": train_data_loader}
    elapsed = time.time() - since
    fabric.print(f"Dataset loaded in {int(elapsed // 60)}m {round(elapsed % 60, 2)}s")

    # Configure output directory
    directory_path = os.path.join(config['out_dir'], config['wandb_name'])
    if fabric.global_rank == 0:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    # Train model
    model = train_model(fabric, data_loaders, model, criterion, optimizer, config["grad_acc"], state["step"], state["best_loss"], directory_path,tokens=tokens)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--from_checkpoint', action='store_true', help='Reset training from scratch')
    parser.add_argument('--extra_epoch', action='store_true', help='Train a whole extra epoch from the checkpoint')
    args = parser.parse_args()
    config_path = args.config

    # Load configuration file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(f"Using configuration file: {config_path}")

    # Run main function
    main(config, args.from_checkpoint, args.extra_epoch)
