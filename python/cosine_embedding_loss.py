import torch
import torch.nn as nn

# Initialize the loss function
# reduction='mean' averages the loss across the batch
cosine_loss_fct = nn.CosineEmbeddingLoss(reduction='mean')

def calculate_cosine_loss(student, teacher, inputs, masks, device='cuda'):
    # 1. Forward pass with output_hidden_states=True
    # We use torch.no_grad() for the teacher because we aren't updating it
    with torch.no_grad():
        teacher_outputs = teacher(inputs, attention_mask=masks, output_hidden_states=True)
    
    student_outputs = student(inputs, attention_mask=masks, output_hidden_states=True)

    # 2. Extract Hidden States
    # The output is a tuple: (embeddings, layer_1, layer_2, ..., layer_N)
    # We usually skip the 0th element (embeddings) and focus on the Transformer layers
    teacher_hiddens = teacher_outputs.hidden_states[1:] 
    student_hiddens = student_outputs.hidden_states[1:]
    
    total_cosine_loss = 0.0
    
    # 3. Iterate through Student layers
    # The Student has 6 layers. We map Student layer 'i' to Teacher layer '2*i + 1'
    # (Assuming we initialized from layers 1, 3, 5... which correspond to indices 0, 2, 4 in 0-indexed lists)
    # NOTE: The exact mapping depends on initialization. If you initialized from 0, 2, 4, use 2*i.
    
    for i in range(len(student_hiddens)):
        # Select the corresponding teacher layer (Skip connection logic)
        teacher_layer_idx = (i * 2) + 1 
        
        s_hidden = student_hiddens[i]
        t_hidden = teacher_hiddens[teacher_layer_idx]
        
        # 4. Flatten the tensors for the Loss Function
        # Shape becomes: (batch_size * sequence_length, hidden_dim)
        # This treats every token in every sentence as a separate vector to align
        s_hidden_flat = s_hidden.view(-1, s_hidden.size(-1))
        t_hidden_flat = t_hidden.view(-1, t_hidden.size(-1))
        
        # 5. Create Target Tensor (1.0 means "make these vectors similar")
        target = torch.ones(s_hidden_flat.size(0)).to(device)
        
        # 6. Calculate Loss for this layer pair
        loss = cosine_loss_fct(s_hidden_flat, t_hidden_flat, target)
        total_cosine_loss += loss

    return total_cosine_loss

# Hyperparameters
alpha_ce = 5.0      # MLM Loss weight
alpha_distil = 2.0  # KL Divergence weight
alpha_cos = 1.0     # Cosine Loss weight (Usually lower than MLM)

# Inside your training loop:
loss_cos = calculate_cosine_loss(student, teacher, inputs, masks)

# Final combined loss
total_loss = (alpha_ce * loss_mlm) + (alpha_distil * loss_distill) + (alpha_cos * loss_cos)
