import torch

sentence_transformer_model = "all-MiniLM-L6-v2"
sample_sentences = ["Her brother is a king.",
                     "His sister is a queen.",
                     "They can't dance."]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 30