from transformers import EsmTokenizer, EsmModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)

# Generate dummy protein sequences of length ~400
batch_size = 32
sequences = ["M" * 400 for _ in range(batch_size)]

inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

print(outputs.last_hidden_state.shape)  # Should be (batch_size, seq_len, hidden_size)
