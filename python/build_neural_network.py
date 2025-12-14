import torch

# 1. DATA
# Input: 3 numbers. Target: The sum of those numbers.
# Example: Input [1, 2, 3] -> Target [6]
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [10.0, 0.0, -5.0]])
y = torch.tensor([[6.0], [15.0], [5.0]])

# 2. ARCHITECTURE (The Blueprint)
input_size = 3   # We take 3 numbers in
hidden_size = 4  # We have 4 "neurons" in the middle layer to think about it
output_size = 1  # We output 1 number (the sum)

# 3. INITIALIZE WEIGHTS (Random Garbage)
# W1 connects Input -> Hidden
# W2 connects Hidden -> Output
# requires_grad=True tells PyTorch: "Track the math we do on these so we can fix them later"
W1 = torch.randn(input_size, hidden_size, requires_grad=True) 
b1 = torch.randn(hidden_size, requires_grad=True)             

W2 = torch.randn(hidden_size, output_size, requires_grad=True)
b2 = torch.randn(output_size, requires_grad=True)

print(f"Initial Random Weight (W1 top left): {W1[0][0].item():.4f}")

learning_rate = 0.01  # How big of a change we make to weights each time

for epoch in range(1000): # Repeat 1000 times
    
    # --- STEP 2: FORWARD PASS (The Prediction) ---
    # Manual Matrix Multiplication: Input @ Weights + Bias
    
    # Layer 1
    hidden_layer_input = X @ W1 + b1
    hidden_layer_output = torch.relu(hidden_layer_input) # Activation function (removes negatives)
    
    # Layer 2 (Final Prediction)
    final_prediction = hidden_layer_output @ W2 + b2
    
    # --- STEP 3: LOSS CALCULATION (The Error) ---
    # Mean Squared Error: (Prediction - Real Answer)^2
    loss = (final_prediction - y).pow(2).mean()
    
    # --- STEP 4: BACKPROPAGATION (Finding the Blame) ---
    # This calculates the gradient (derivative) for W1, b1, W2, b2
    loss.backward()
    
    # --- OPTIMIZATION (Updating the Weights) ---
    with torch.no_grad(): # Pause gradient tracking to change weights
        # Move weights in the opposite direction of the gradient
        W1 -= learning_rate * W1.grad
        b1 -= learning_rate * b1.grad
        W2 -= learning_rate * W2.grad
        b2 -= learning_rate * b2.grad
        
        # Reset gradients to zero for the next loop
        W1.grad.zero_()
        b1.grad.zero_()
        W2.grad.zero_()
        b2.grad.zero_()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

# Final Test
print("\n--- Training Complete ---")
print(f"Target: 6.0 | Prediction: {final_prediction[0].item():.4f}")
