use candle_core::{Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};
use std::fmt;

pub struct SimpleNet {
    layer1: Linear,
    layer2: Linear,
}

impl SimpleNet {
    // This function initializes the weights "from scratch" (randomly)
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let layer1 = candle_nn::linear(3, 4, vs.pp("layer1"))?; // Input 3 -> Hidden 4
        let layer2 = candle_nn::linear(4, 1, vs.pp("layer2"))?; // Hidden 4 -> Output 1
        Ok(Self { layer1, layer2 })
    }

    // Forward pass
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x -> Layer 1 -> ReLU -> Layer 2 -> Output
        let x = self.layer1.forward(x)?;
        let x = x.relu()?;
        let x = self.layer2.forward(&x)?;
        Ok(x)
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone)]
pub enum SentimentSentence {
    O,
    BLook,
    ILook,
    BMove,
    IMove,
    BTurn,
    ITurn,
}

impl fmt::Display for SentimentSentence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SentimentSentence::O => write!(f, "O"),
            SentimentSentence::BLook => write!(f, "BLook"),
            SentimentSentence::ILook => write!(f, "ILook"),
            SentimentSentence::BMove => write!(f, "BMove"),
            SentimentSentence::IMove => write!(f, "IMove"),
            SentimentSentence::BTurn => write!(f, "BTurn"),
            SentimentSentence::ITurn => write!(f, "ITurn"),
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone)]
pub enum Sentiment {
    Negative,
    Positive,
    Unknown,
}

impl fmt::Display for Sentiment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sentiment::Negative => write!(f, "Negative"),
            Sentiment::Positive => write!(f, "Positive"),
            Sentiment::Unknown => write!(f, "Unknown"),
        }
    }
}
