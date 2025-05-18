import os
import time
from tqdm import tqdm

import torch
import os
from datasets import TOKENS,TOKENS_ESM1



def train_model(fabric, data_loaders, model, criterion, optimizer, grad_acc, step, best_loss, directory_path,tokens):
    fabric.print("Training model from step:", step, "with best loss:", best_loss)
    model.train() # Set model to training mode
    since = time.time() # Start time
    initial_step = step
    best_state = None
    # Iterate over batches
    fabric.barrier()
    for (x, y) in data_loaders['train']:
        mask = y != tokens.index('#') # torch.Size([21, 47])
        y_expected = y[mask]

        # # Forward pass
        # is_accumulating = step % grad_acc != 0
        # with fabric.no_backward_sync(model, enabled=is_accumulating):
        #     logits = model(x)["last_hidden_state"][mask] # torch.Size([21, 47, 2048])
        #     loss = criterion(logits, y_expected)
        #     fabric.backward(loss)
        # if not is_accumulating:
        #     optimizer.step()
        #     optimizer.zero_grad()

        # Forward pass
        
        is_accumulating = step % grad_acc != 0
        step += 1 #Temp movement for tracking, move again later
        logits = model(x)[mask] # torch.Size([21, 47, 2048])
        
        loss = criterion(logits, y_expected)
        #print(loss)
        fabric.backward(loss)
        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()

        # Save model weights at specific steps
        if step in [2000, 4000, 6000, 8000, 10000]:
            save_path = os.path.join(directory_path, f"weights_step_{step}.ckpt")
            state = {"model": model, "optimizer": optimizer, "step": step, "best_loss": best_loss}
            fabric.save(save_path, state)

        # Track the best model based on loss
        if loss.item() < best_loss:
            fabric.print(f"New best loss: {loss.item()}")
            #fabric.print(f"Save started at step : {step}")
            best_loss = loss.item()
            #fabric.save(save_path, best_state)
            #fabric.print(f"Save done at step : {step}")
        # Save checkpoint every 10000 steps (overwrites the previous save)
        if step % 20000 == 0: 
            # Save current model
            save_path = os.path.join(directory_path, "weights_last.ckpt")
            save_path2 = os.path.join(directory_path, f"weights_step_{step}.ckpt")
            current_state = {"model": model, "optimizer": optimizer, "step": step, "best_loss": best_loss}
            fabric.save(save_path, current_state)
            fabric.save(save_path2, current_state)

        # Update metrics
        fabric.log_dict({"step": step, "best loss": best_loss, "loss": loss.item(), "ppl": torch.exp(loss).item()})
        #step += 1
        
    # Save model at the end
    save_path = os.path.join(directory_path, "weights_final.ckpt")
    final_state = {"model": model, "optimizer": optimizer, "step": step, "best_loss": best_loss}
    fabric.save(save_path, final_state)

    # Save best model at the end

    # Report time
    elapsed = time.time() - since
    
    days = elapsed // 86400
    remaining_seconds = elapsed % 86400
    hours = remaining_seconds // 3600
    remaining_seconds %= 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60

    fabric.print(
        f"{step - initial_step} steps completed in {days}d {hours}h {minutes}m {round(seconds, 2)}s"
    )
    
    return model

