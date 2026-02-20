import numpy as np
import matplotlib.pyplot as plt

def get_positional_encoding(seq_len, d_model):
    pos_enc = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return pos_enc

seq_length = 50
model_dim = 128

encoding = get_positional_encoding(seq_length, model_dim)

plt.figure(figsize=(10, 8))
plt.pcolormesh(encoding, cmap='viridis')
plt.xlabel('Embedding Dimension')
plt.ylabel('Token Position')
plt.colorbar(label='Encoding Value')
plt.title("Transformer: Positional Encoding Matrix")
plt.show()

print("Positional Encoding generated successfully!")