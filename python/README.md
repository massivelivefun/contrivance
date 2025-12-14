# export_distilbert.py
https://huggingface.co/distilbert/distilbert-base-uncased
pip install torch transformers onnx onnxscript onnxruntime

# train_commands.py & test_train_commands_output.py
pip install torch transformers scikit-learn pandas

# train_ner.py
pip install torch transformers datasets seqeval

pip install optimum[exporters] onnx onnxruntime transformers[torch] accelerate


py -3.13 ./python/train_ner.py
py -3.13 -m optimum.exporters.onnx --model ./model/my_ner_model ./model/onnx_export --task token-classification

# many agents agentic idea (mobile target vram??)

