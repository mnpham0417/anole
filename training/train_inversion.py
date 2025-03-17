import torch
import deepspeed
import jsonlines

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import ChameleonForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from pathlib import Path

from constants_training import (
    ANOLE_PATH_HF,
    ANOLE_PATH_HF_TRAINED,
    DATASET_TOKENIZED_PATH
)

# concept = "van_gogh"
concept = "dog"
ANOLE_PATH_HF = Path("/scratch/mp5847/workspace/mixed-modal-erasure/src/anole/Anole-7b-v0.1_hf")
ANOLE_PATH_HF_TRAINED = Path(f"/scratch/mp5847/workspace/mixed-modal-erasure/src/anole/Anole-7b-v0.1_hf_trained_{concept}_<art>")
DATASET_TOKENIZED_PATH = Path(f"/scratch/mp5847/workspace/mixed-modal-erasure/src/anole/training/data/{concept}/tokenized_metadata.jsonl")

# Define the dataset class
class TokenizedDataset(Dataset):
    def __init__(self, filepath):
        self.tokenized_data = []
        with jsonlines.open(filepath) as reader:
            for obj in reader:
                self.tokenized_data.append(torch.tensor(obj['text_tokens'] + obj['image_tokens'], dtype=torch.long))
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx],

# Define custom collate function for DataLoader
def collate_fn(batch):
    batch_inputs = [item[0] for item in batch]
    batch_inputs_padded = pad_sequence(batch_inputs, batch_first=True, padding_value=-100)

    # Create attention masks
    attention_masks = torch.zeros_like(batch_inputs_padded, dtype=torch.long)
    attention_masks = attention_masks.masked_fill(batch_inputs_padded != -100, 1)
   
    return {'input_ids': batch_inputs_padded, 'attention_mask': attention_masks, 'labels': batch_inputs_padded.clone()}

# Custom Trainer class to update only specific embeddings
class TextualInversionTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Store original embeddings that should not be updated
        # Make sure to store on the same device as the model
        self.original_embeddings = self.model.get_input_embeddings().weight.data.clone().to(self.model.device)
        
        # Create a mask for tokens that should not be updated
        self.mask_no_updates = torch.ones((len(self.original_embeddings),), dtype=torch.bool, device=self.model.device)
        self.mask_no_updates[8197:] = False
    
    def training_step(self, model, inputs):
        # Run normal training step
        loss = super().training_step(model, inputs)
        
        # After the optimizer step, restore embeddings for tokens that should not be updated
        with torch.no_grad():
            current_embeddings = model.get_input_embeddings().weight.data
            current_embeddings[self.mask_no_updates] = self.original_embeddings[self.mask_no_updates]
            
            # Also restore LM head weights for tokens that should not be updated if they share weights
            # if hasattr(model, 'lm_head') and model.get_output_embeddings() is not None:
            #     model.lm_head.weight.data[self.mask_no_updates] = self.original_embeddings[self.mask_no_updates]
        # #find index of the token embedding that are updated in dimension 0
        # updated_embeddings = model.get_input_embeddings().weight.data
        # # Check if any element in a row is different
        # row_changed = torch.any(updated_embeddings != self.original_embeddings, dim=1)
        # # Get the indices of rows that changed
        # changed_indices = torch.where(row_changed)[0]
        # print(changed_indices, len(changed_indices))
        # #get a list of indices that are greater than 8192
        # changed_indices_list = changed_indices[changed_indices > 8197].tolist()
        # print(changed_indices_list)
        return loss

# Initialize the model
model = ChameleonForCausalLM.from_pretrained(ANOLE_PATH_HF).to("cuda")
print(model)

#add lora

# peft_config = LoraConfig(
#             r=16,
#             lora_alpha=16,
#             lora_dropout=0.1,
#             target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj","gate_proj"],
#             task_type="CAUSAL_LM",
#         )
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()

# assert 0
#freeze all layers except embed_tokens
for name, param in model.named_parameters():
    if "embed_tokens" not in name:
        param.requires_grad = False
        
#print all trainable parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

# Initialize the dataset
dataset = TokenizedDataset(DATASET_TOKENIZED_PATH)

# Define training arguments
training_args = TrainingArguments(
    output_dir=ANOLE_PATH_HF_TRAINED,
    learning_rate=1e-3,
    num_train_epochs=1500,
    per_device_train_batch_size=1,
    save_steps=3000,
    fp16=False,
    # fp16=True,
    logging_strategy="steps",
    logging_steps=1,  # Log every 1 steps
    deepspeed="ds_config.json"
)

# Initialize the Trainer with custom collate_fn
trainer = TextualInversionTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn
)

# Train the model
trainer.train()

# Save the model
torch.save(model.state_dict(), ANOLE_PATH_HF_TRAINED / 'pytorch_model.bin')
