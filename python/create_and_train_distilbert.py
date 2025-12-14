import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForMaskedLM, DistilBertForMaskedLM, DistilBertConfig

# 1. Load the Teacher (Pre-trained BERT)
teacher = BertForMaskedLM.from_pretrained('bert-base-uncased')
teacher.eval()  # Freeze teacher weights (we don't train the teacher)

# 2. Initialize the Student (DistilBERT)
# We create a configuration with 6 layers instead of 12
config = DistilBertConfig(
    n_layers=6, 
    vocab_size=teacher.config.vocab_size, 
    hidden_dim=teacher.config.hidden_size,
    dim=teacher.config.hidden_size, # Keep hidden size same (768)
    n_heads=teacher.config.num_attention_heads
)
student = DistilBertForMaskedLM(config)

# 3. The "Sanity" Initialization (Crucial Step)
# Copy weights from Teacher layers 0, 2, 4, 6, 8, 10 to Student layers 0, 1, 2, 3, 4, 5
teacher_layers = teacher.bert.encoder.layer
student_layers = student.distilbert.transformer.layer

for i in range(6):
    student_layers[i].load_state_dict(teacher_layers[2 * i].state_dict())

# Copy embeddings as well
student.distilbert.embeddings.load_state_dict(teacher.bert.embeddings.state_dict())

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
    Calculates KL Divergence between softened Teacher and Student distributions.
    """
    # Soften probabilities with Temperature T
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
    
    # Calculate KL Divergence
    return nn.KLDivLoss(reduction="batchmean")(soft_prob, soft_targets) * (temperature ** 2)

# --- Simulated Training Step ---
optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5)
temperature = 2.0
alpha_ce = 5.0  # Weight for MLM loss
alpha_distil = 2.0 # Weight for Distillation loss

# Assume 'inputs' is a batch of tokenized text with 'input_ids' and 'attention_mask'
def training_step(batch):
    inputs = batch['input_ids']
    masks = batch['attention_mask']
    
    # 1. Get Teacher Output (No Gradient Calculation)
    with torch.no_grad():
        teacher_outputs = teacher(inputs, attention_mask=masks)
        teacher_logits = teacher_outputs.logits

    # 2. Get Student Output
    student_outputs = student(inputs, attention_mask=masks, labels=inputs)
    student_logits = student_outputs.logits
    
    # 3. Calculate Losses
    
    # Loss A: Masked Language Modeling (Standard Cross Entropy)
    # The Hugging Face model calculates this internally if 'labels' are provided
    loss_mlm = student_outputs.loss 
    
    # Loss B: Distillation Loss (KL Divergence)
    loss_distill = distillation_loss(student_logits, teacher_logits, temperature)
    
    # Loss C: Cosine Embedding Loss (Optional but recommended in original paper)
    # Checks alignment of hidden states (omitted here for brevity, requires hooking into hidden layers)

    # 4. Combine Losses
    total_loss = (alpha_ce * loss_mlm) + (alpha_distil * loss_distill)
    
    # 5. Backpropagation
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return total_loss.item()