#[allow(dead_code)]
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
    logits.iter().map(|&x| (x - max_logit).exp() / exp_sum).collect()
}
