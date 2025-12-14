use ort::tensor::Shape;
use ort::session::Session;
use ort::session::builder::{GraphOptimizationLevel};
use ort::value::Tensor;
use ort::value::ValueRef;
use tokenizers::Tokenizer;
use ndarray::Array2;
use ndarray::ArrayView2;
use std::error::Error;

// "0": "O",
// "1": "B-LOOK",
// "2": "I-LOOK",
// "3": "B-MOVE",
// "4": "I-MOVE",
// "5": "B-TURN",
// "6": "I-TURN"

// bugged words
// strafe, meme

// Order matter which is a fucked up bug need to fix that
pub const LABELS: &[&str] = &["O", "B-LOOK", "I-LOOK", "B-MOVE", "I-MOVE", "B-TURN", "I-TURN"];

pub struct Model {
    tokenizer: Box<Tokenizer>,
    session: Session,
}

pub fn init_model() -> Result<Model, Box<dyn Error>> {
    // Setup model
    let tokenizer = Box::new(Tokenizer::from_file("./model/onnx_export/tokenizer.json").unwrap());

    let model_path = "./model/onnx_export/model.onnx";
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    // println!("Model loaded successfully");

    Ok(Model {
        tokenizer,
        session,
    })
}

impl Model {
    pub fn compute_input_text(&mut self, input_text: &str) -> Result<Vec<String>, Box<dyn Error>> {
        // let input_text = match 0 {
        //     0 => "Our comrades have been slain.",
        //     1 => "The treasure chest is locked tightly.",
        //     _ => "Meow."
        // };

        let encoding = Box::new(self.tokenizer.encode(input_text, true).unwrap());

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        let seq_len = input_ids.len();

        let input_ids_array = Array2::from_shape_vec((1, seq_len), input_ids)?;
        let mask_array = Array2::from_shape_vec((1, seq_len), attention_mask)?;

        let input_tensor = Tensor::from_array(input_ids_array)?;
        let mask_tensor = Tensor::from_array(mask_array)?;

        // Run the model
        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_tensor,
            "attention_mask" => mask_tensor,
        ])?;

        // Read the output
        let output_tensor: (&Shape, &[f32]) = outputs["logits"].try_extract_tensor::<f32>()?;
        // let view = outputs["logits"].view();
        // let view = output_tensor.view();
        // let output_shape = output_tensor.shape();
        // let num_labels = output_shape[2];

        let dims = output_tensor.0;
        let raw_data = output_tensor.1;

        // should be 3 dimensional
        let seq_len = dims[1] as usize;
        let num_labels = dims[2] as usize;

        let data = ArrayView2::from_shape((seq_len, num_labels), raw_data)
            .expect("Shape mismatch between ONNX output and ndarray view");

        let tokens = encoding.get_tokens();

        let mut results = Vec::new();
        results.push(format!("--- Results ---"));
        // println!("\n--- Results ---");

        for i in 0..seq_len {
            let token = &tokens[i];

            if token == "[CLS]" {
                // results.push(format!("[CLS]"));
                continue;
            }
            
            if token == "[SEP]" {
                // results.push(format!("[SEP]"));
                continue;
            }

            let logits = data.row(i);

            let max_logit = logits.fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            let mut exp_sum = 0.0;
            let _exps: Vec<f32> = logits.iter().map(|&x| {
                let e = (x - max_logit).exp();
                exp_sum += e;
                e
            }).collect::<Vec<_>>();
            
            // let mut best_score = f32::NEG_INFINITY;
            let mut best_prob = 0.0;
            let mut best_label_idx = 0;
            
            for (label_idx, &exp_val) in data.row(i).iter().enumerate() {
                let prob = exp_val / exp_sum;
                if prob > best_prob {
                // if score > best_score {
                    // best_score = score;
                    best_prob = prob;
                    best_label_idx = label_idx;
                }
            }

            let label_name = LABELS[best_label_idx];

            // if label_name != "O" {
                let clean_token = token.replace("##", "");
                // println!("Word {:<15} -> Label: {}", clean_token, label_name);
                results.push(format!("Word: {:<15} -> Label: {:<10} ({:.1}%)", clean_token, label_name, best_prob * 10.0));
            // }
        }

        // Self::debug_model(&view, output_tensor.1);

        // let logits_slice = output_tensor.1;

        // let probabilities = softmax(logits_slice);

        // println!("Negative: {:.2}%", probabilities[0] * 100.0);
        // println!("Positive: {:.2}%", probabilities[1] * 100.0);

        // let max_index = probabilities.iter().enumerate()
        //     .max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(index, _)| index);

        // match max_index {
        //     Some(0) => Result::Ok((Sentiment::Negative, probabilities[0])),
        //     Some(1) => Result::Ok((Sentiment::Positive, probabilities[1])),
        //     _ => Result::Ok((Sentiment::Unknown, 0.0)),
        // }

        // match max_index {
        //     Some(0) => Result::Ok((SentimentSentence::O, probabilities[0])),
        //     Some(1) => Result::Ok((SentimentSentence::B_LOOK, probabilities[1])),
        //     Some(2) => Result::Ok((SentimentSentence::I_LOOK, probabilities[2])),
        //     Some(3) => Result::Ok((SentimentSentence::B_MOVE, probabilities[3])),
        //     Some(4) => Result::Ok((SentimentSentence::I_MOVE, probabilities[4])),
        //     Some(5) => Result::Ok((SentimentSentence::B_TURN, probabilities[5])),
        //     Some(6) => Result::Ok((SentimentSentence::I_TURN, probabilities[6])),
        //     _ => Result::Ok((SentimentSentence::O, 0.0)),
        // }

        // println!("Sentiment: {}", sentiment);

        Ok(results)
    }

    #[allow(dead_code)]
    pub fn debug_model(view: &ValueRef<'_>, second_tensor_element: &[f32]) {
        println!("Logits: {:?}\n", view);

        let (max_idx, max_val) = second_tensor_element
            .iter()
            .enumerate()
            .fold((0, f32::MIN), |(i_max, v_max), (i, &v)| {
                if v > v_max { (i, v) } else { (i_max, v_max) }
            });

        println!("Predicted Class ID: {} (Score: {})", max_idx, max_val);
    }
}
