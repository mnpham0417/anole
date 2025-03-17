import os
import torch
from pathlib import Path
import argparse
from transformers import ChameleonForCausalLM

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory."""
    checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')]
    if not checkpoints:
        return None
    
    # Sort by checkpoint number (e.g., checkpoint-1000 comes after checkpoint-500)
    latest_checkpoint = max(checkpoints, key=lambda d: int(d.name.split('-')[1]))
    return latest_checkpoint

def load_checkpoint_and_save_model(checkpoint_dir, output_path):
    """
    Load the latest checkpoint and save it to pytorch_model.bin.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        output_path: Path to save the final model
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint:
        print(f"Loading latest checkpoint: {latest_checkpoint}")
        
        # Load the model from checkpoint
        model = ChameleonForCausalLM.from_pretrained(latest_checkpoint)
        
        # Save properly
        torch.save(model.state_dict(), output_path)
        print(f"Model saved to {output_path}")
        return True
    else:
        print("No checkpoints found.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load checkpoint and save model")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory containing checkpoints")
    parser.add_argument("--output_path", default="pytorch_model.bin", help="Path to save the model")
    args = parser.parse_args()
    
    load_checkpoint_and_save_model(args.checkpoint_dir, args.output_path)