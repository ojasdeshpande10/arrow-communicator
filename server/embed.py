from typing import List

from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
import time
class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize the text
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {key: val.squeeze(0) for key, val in inputs.items()}
class Embedder:
    def __init__(self, model_name_or_path:str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        if torch.cuda.is_available():
            self.model.to("cuda")
            self.model = self.model.half()
    
    def embed(self, texts:List[str]):
        # Fill in code to generate embeddings
        # inputs = self.tokenizer(text, padding="longest", truncation=True, return_tensors="pt",max_length=512)
        start_time_tokenizing = time.time()
        dataset = TextDataset(texts, self.tokenizer)
        end_time_tokenizing = time.time()
        start_time_embedding = time.time()
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        all_embeddings=[]
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to("cuda") for k, v in batch.items()}
                outputs = self.model(**batch)
                hidden_states = outputs.hidden_states
                last_four_layers = hidden_states[-4:]
                # taking the mean across the sequence length
                mean_last_four_layers = [layer.mean(dim=1) for layer in last_four_layers]
                # stack the mean embeddings from last four layers
                pooled_embeddings = torch.cat(mean_last_four_layers,dim=-1) # Shape (batch_size, 4*1024)
                pooled_embeddings = pooled_embeddings.cpu().tolist()
                all_embeddings.extend(pooled_embeddings)
        end_time_embedding = time.time()
        return all_embeddings, (end_time_tokenizing-start_time_tokenizing), (end_time_embedding-start_time_embedding)