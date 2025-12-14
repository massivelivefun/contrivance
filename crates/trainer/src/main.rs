use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{Optimizer, VarMap, VarBuilder};
use model_lib::SimpleNet;

fn main() -> Result<()> {
    // Use CPU (or Device::new_cuda(0) for GPU)
    let device = Device::Cpu;

    // --- DATA ---
    // Input: [1,2,3], [4,5,6], [10,0,-5]
    let x_data = Tensor::new(&[
        [1.0f32, 2.0, 3.0], 
        [4.0, 5.0, 6.0], 
        [10.0, 0.0, -5.0]
    ], &device)?;
    
    // Target: [6], [15], [5]
    let y_data = Tensor::new(&[[6.0f32], [15.0], [5.0]], &device)?;

    // --- INITIALIZATION ---
    // 'VarMap' holds the weights and handles the "gradients" (the math history)
    let varmap = VarMap::new(); 
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    // Create the model with random weights
    let model = SimpleNet::new(vs)?;

    // Create the Optimizer (AdamW is standard)
    // This replaces "W1 -= learning_rate * grad" with a smarter update rule
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), candle_nn::ParamsAdamW {
        lr: 0.1, // Learning Rate
        ..Default::default()
    })?;

    println!("--- Starting Training ---");

    // --- TRAINING LOOP ---
    for epoch in 0..1000 {
        // Step 2: Forward Pass
        let predictions = model.forward(&x_data)?;

        // Step 3: Loss Calculation (Mean Squared Error)
        let loss = (predictions.sub(&y_data)?.sqr()?.mean_all())?;

        // Step 4: Backpropagation & Optimization
        // 'backward()' calculates all the gradients automatically
        optimizer.backward_step(&loss)?;

        if epoch % 100 == 0 {
            println!("Epoch {}: Loss {}", epoch, loss.to_scalar::<f32>()?);
        }
    }

    // --- FINAL TEST ---
    println!("\n--- Training Complete ---");
    let final_pred = model.forward(&x_data)?;
    println!("Target: 6.0 | Prediction: {:.4}", final_pred.get(0)?.get(0)?.to_scalar::<f32>()?);
    varmap.save("weights.safetensors")?;
    println!("Model trained and saved!");

    // --- 3. LOAD THE MODEL (In a real app, this would be a separate program) ---
    println!("\nLoading model from file...");

    // A. Create a NEW VarMap (empty)
    let mut loaded_varmap = VarMap::new();
    
    // B. Load the data from the file INTO this new VarMap
    loaded_varmap.load("weights.safetensors")?;

    // C. Create a VarBuilder from this loaded map
    // exact same structure as before, but now it has the trained numbers
    let loaded_vs = VarBuilder::from_varmap(&loaded_varmap, DType::F32, &device);

    // D. Re-build the architecture
    let loaded_model = SimpleNet::new(loaded_vs)?;

    // --- 4. VERIFY IT WORKS ---
    let test_input = Tensor::new(&[[1.0f32, 2.0, 3.0]], &device)?;
    let result = loaded_model.forward(&test_input)?;
    
    println!("Loaded Model Prediction: {:.4} (Should be close to 6.0)", result.get(0)?.get(0)?.to_scalar::<f32>()?);

    Ok(())
}
