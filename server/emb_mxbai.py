from typing import List, Dict, Optional
import torch, numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import torch.nn.functional as F

class TextDataset(Dataset):

    def __init__(self, texts: List[str], tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.model_max_length
        self.truncated_count = 0  # Counter for truncated messages
        self.short_indices = []
        self.shorter_texts = []
        self.longer_texts = []
        self.long_indices = [] 
        self._raw_input_ids = []
        self._raw_attention_masks = []
        for idx, text in enumerate(texts):
            # First tokenize without truncation to check length
            tokenized_output_no_trunc = self.tokenizer(text, add_special_tokens=False, truncation=False)
            L = len(tokenized_output_no_trunc['input_ids'])
            # Check if truncation is required
            if L <= self.max_length-2:
                self.short_indices.append(idx)
                self.shorter_texts.append(text)
            else:
                self.long_indices.append(idx)
                self.longer_texts.append(text)
                self._raw_input_ids.append(tokenized_output_no_trunc['input_ids']) 
                self._raw_attention_masks.append(tokenized_output_no_trunc['attention_mask'])
        print(f"Number of longer messages: {len(self.long_indices)}")



class UniversalEmbedder:
    """
    Universal Embedder for Mixedbread and Sentence-Transformers models
    ------------------------------------------------------------
    - Supports long context by chunking.
    - Uses mean pooling.
    - Supports truncation for mixedbread if needed.

    Parameters
    ----------
    model_name : str
    trunc_dim : Optional[int]  (Only used for Mixedbread)
    device : Optional[str]
    """

    def __init__(self, model_name: str, trunc_dim: Optional[int] = 256, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model_name = model_name.lower()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.longer_sentences = 0
        self.len_longest = 0
        self.len_longer_256 =0
        if self.device.type == "cuda":
            self.model = self.model.half()
        self.model.eval()

        # Model-specific settings
        if "mixedbread" in self.model_name:
            self.model_type = "mixedbread"
            if not (1 <= trunc_dim <= 1024):
                raise ValueError("trunc_dim must be between 1 and 1024 for Mixedbread.")
            self.trunc_dim = trunc_dim
            self.max_tokens = 512
        else:
            self.model_type = "sentence-transformers"
            self.trunc_dim = None
            self.max_tokens = self.tokenizer.model_max_length  # usually 512
            print(f"Using {self.model_type} model with max tokens: {self.max_tokens}")


    def collate_tokenize(self, batch):
        '''
        collates and tokenizes the batch
        '''
        return self.tokenizer(batch, padding="longest", truncation=True, return_tensors="pt", max_length=self.max_tokens)
    @torch.inference_mode()
    def embed(self, texts: List[str], batch_size: int = 256, progress: bool = True):
        """
        Return {"embeddings": List[List[float]]} — shape: (len(texts), dim)
        Handles long inputs by chunking and aggregation.
        """
        all_embeddings = []
        self.longer_sentences = 0
        self.len_longest = 0
        dataset = TextDataset(texts, self.tokenizer)
        dataloader = DataLoader(dataset.shorter_texts, batch_size=64, shuffle=False, collate_fn=self.collate_tokenize)
        shorter_embeddings = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = {k: v.to("cuda") for k, v in batch.items()}
                outputs = self.model(**batch)
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = batch["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                sentence_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                sentence_embeddings_trans = F.normalize(sentence_embedding, p=2, dim=1)
                shorter_embeddings.extend(sentence_embedding.cpu().tolist())
        longer_embeddings = []
        # Process longer sentences
        if len(dataset.longer_texts) > 0:
                chunks = []
                chunk_size = self.max_tokens - 2  # to account for [CLS] and [SEP]
                long_input_ids = dataset._raw_input_ids
                long_attention_masks = dataset._raw_attention_masks
                
                for input_ids, attention_mask in zip(long_input_ids, long_attention_masks):
                    chunks = []
                    chunk_size = self.max_tokens - 2  # to account for [CLS] and [SEP]
                    for i in range(0, len(input_ids), chunk_size):
                        chunk_input_ids = [self.tokenizer.cls_token_id] + input_ids[i:i + chunk_size] + [self.tokenizer.sep_token_id]
                        chunk_attention_mask = [1] + attention_mask[i:i + chunk_size] + [1]

                        chunks.append({
                            "input_ids": torch.tensor(chunk_input_ids, device=self.device).unsqueeze(0),
                            "attention_mask": torch.tensor(chunk_attention_mask, device=self.device).unsqueeze(0)
                        })

                chunk_embeddings = []
                for chunk in chunks:
                    outputs = self.model(**chunk)
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = chunk["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                    embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    embedding = F.normalize(embedding, p=2, dim=1)
                    chunk_embeddings.append(embedding.squeeze(0))

                message_embedding = torch.stack(chunk_embeddings).mean(dim=0)
                longer_embeddings.append(message_embedding.cpu().tolist())
        N = len(texts)
        # prepare an output list of the right size
        embs = [None] * N

        # write the short embeddings into place
        for idx, emb in zip(dataset.short_indices, shorter_embeddings):
            embs[idx] = emb

        # write the long embeddings into place
        for idx, emb in zip(dataset.long_indices, longer_embeddings):
            embs[idx] = emb

        result = {"embeddings": embs}
        return result


# ------------------- Example Usage --------------------
if __name__ == "__main__":
    sentences = ["Huxley seems to be saying the following: *Because we cannot transmit our experiences directly from one brain to another, we are profoundly alone. * \n I would argue that this is precisely why we need and want each other. We meet up with friends, share opinions, look at each others' photos, and perhaps share emotional connections as well. We make ourselves vulnerable to others so that we can relate more deeply. We do this deliberately for the purpose of understanding. \n If we really could transmit exactly what we were thinking, feeling, and sensing at all times, why would we ever need to talk to each other? Why would we ever ask questions? If we could really do what Huxley is describing, I think we would have a real end to human communication and relationship."]

    # Mixedbread Example
    # embedder = UniversalEmbedder(model_name="mixedbread-ai/mxbai-embed-large-v1", trunc_dim=512)
    # res = embedder.embed(long_sentences)
    # print("Mixedbread:", len(res["embeddings"]), "vectors of dim", len(res["embeddings"][0]))

    # Sentence-Transformers Example
    # embedder2 = UniversalEmbedder(model_name="sentence-transformers/all-MiniLM-L12-v2")
    # res2 = embedder2.embed(sentences)
    # print("MiniLM:", len(res2["embeddings"]), "vectors of dim", len(res2["embeddings"][0]))
    # print("EMBEDDING:", res2["embeddings"])
    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    st_model = SentenceTransformer(model_name)
    sentence_embeddings = st_model.encode(sentences)
    print("EMBEDDING:", sentence_embeddings)
