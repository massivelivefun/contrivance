from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer
import shutil
import os
import onnx

model_path = "./model/my_ner_model"
output_path = "./model/onnx_export"

model = ORTModelForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# model.save_pretrained(output_path)
model.save_pretrained(
    output_path,
    save_as_external_path=True,
    all_tensors_to_one_file=True,
    external_data_format="weights.onnx.data",
)
tokenizer.save_pretrained(output_path)

onnx_model = onnx.load(output_path + "/model.onnx")
final_onnx_path = os.path.join(output_path, "model.onnx")
data_file_name = "model.onnx.data"

onnx.save_model(
    onnx_model,
    final_onnx_path,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=data_file_name,
    size_threshold=1024,
    convert_attribute=False,
)

print("Saved to ", final_onnx_path)
