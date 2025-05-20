from typing import List
from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
import time
import sys
from tqdm import tqdm


class TextDataset(Dataset):

    def __init__(self, texts: List[str], tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
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
    




class Embedder:
    '''

    The Embedder class is designed to tokenize input texts and generate embeddings using a pretrained language model.
    It also provides timing information for different stages of the embedding process, including tokenization,
    data loading, embedding generation, and aggregation.

    Attributes:
        tokenizer (AutoTokenizer): The tokenizer used for text preprocessing.
        model (AutoModel): The pretrained model used for generating embeddings.

    Methods:
        __init__(model_name_or_path: str): Initializes the Embedder with a specified model.
        embed(texts: List[str], layers_to_use: List[int]): Tokenizes the input texts, generates embeddings using
            specified layers of the model, and returns the embeddings along with timing information.
    
    '''
    def __init__(self, model_name_or_path:str):
        self.device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, output_hidden_states=True)
        if torch.cuda.is_available():
            device = torch.device("cuda:0") 
            self.model.to(device)
            self.model = self.model.half()

    def embed_longer_v2(self, dataset, layers_to_use):

        long_input_ids = dataset._raw_input_ids
        long_attention_masks = dataset._raw_attention_masks
        longer_embeddings = []
        for input_ids, attention_mask in zip(long_input_ids, long_attention_masks):
            chunks = []
            chunk_size = 512 - 2  # to account for [CLS] and [SEP]
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
                hidden_states = outputs.hidden_states

                selected_layers = [hidden_states[i] for i in layers_to_use]

                stacked_layers = torch.stack(selected_layers, dim=0) # (4, batch_size, seq_len, hidden_size)
                attention_mask = chunk['attention_mask']
                expanded_attention_mask = attention_mask.unsqueeze(0).unsqueeze(-1) # (1, batch_size, seq_len, 1)



                # Filtering the masked tokens (padded tokens)
                masked_layers = stacked_layers * expanded_attention_mask

                sum_masked_layers = masked_layers.sum(dim=2)  # shape: (4, batch_size, hidden_size)

                valid_tokens_count = expanded_attention_mask.sum(dim=2)  # shape: (4, batch_size, 1); Last dimension contains num of valid tokens for each document

                mean_layers = sum_masked_layers / valid_tokens_count  # shape: (4, batch_size, hidden_size)

                pooled_embeddings = mean_layers.permute(1, 0, 2) # shape (batch, 4, hidden_size)
                chunk_embeddings.append(pooled_embeddings.squeeze(0))
            message_embedding = torch.stack(chunk_embeddings).mean(dim=0)
            longer_embeddings.append(message_embedding.cpu().tolist()) 
        
        return longer_embeddings
    def collate_tokenize(self, batch):
        '''
        collates and tokenizes the batch
        '''
        return self.tokenizer(batch, padding="longest", truncation=True, return_tensors="pt", max_length=512)
    def embed(self, texts:List[str], layers_to_use:List[int]):
        result = {}
        # tokenizing the texts 
        start_time_tokenizing = time.time()
        dataset = TextDataset(texts, self.tokenizer)
        end_time_tokenizing = time.time()

        # creating batches in the dataloading with 
        data_loading_time_start = time.time()
        dataloader = DataLoader(dataset.shorter_texts, batch_size=64, shuffle=False, collate_fn=self.collate_tokenize)
        data_loading_time_end = time.time()

        embedding_generation_time = 0
        embedding_aggregation_time = 0
        shorter_embeddings=[]
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = {k: v.to("cuda") for k, v in batch.items()}
                start_embedding_generation_time = time.time()
                outputs = self.model(**batch)
                end_embedding_generation_time = time.time()

                # Logging embedding generation 
                embedding_generation_time += (end_embedding_generation_time-start_embedding_generation_time)
                
                # Check for hidden_states to be None

                start_embedding_aggregation_time = time.time()
                hidden_states = outputs.hidden_states

                selected_layers = [hidden_states[i] for i in layers_to_use]

                stacked_layers = torch.stack(selected_layers, dim=0) # (4, batch_size, seq_len, hidden_size)
                attention_mask = batch['attention_mask'].to("cuda")
                expanded_attention_mask = attention_mask.unsqueeze(0).unsqueeze(-1) # (1, batch_size, seq_len, 1)



                # Filtering the masked tokens (padded tokens)
                masked_layers = stacked_layers * expanded_attention_mask

                sum_masked_layers = masked_layers.sum(dim=2)  # shape: (4, batch_size, hidden_size)

                valid_tokens_count = expanded_attention_mask.sum(dim=2)  # shape: (4, batch_size, 1); Last dimension contains num of valid tokens for each document

                mean_layers = sum_masked_layers / valid_tokens_count  # shape: (4, batch_size, hidden_size)

                pooled_embeddings = mean_layers.permute(1, 0, 2) # shape (batch, 4, hidden_size)
                # stack the mean embeddings from last four layers
                # pooled_embeddings = torch.stack(mean_last_four_layers, dim=1) # shape (batch,4,1024)
                pooled_embeddings = pooled_embeddings.cpu().tolist()
                shorter_embeddings.extend(pooled_embeddings)
                end_embedding_aggregation_time = time.time()

                # Logging time for embedding aggregation
                embedding_aggregation_time += (end_embedding_aggregation_time-start_embedding_aggregation_time)
                torch.cuda.empty_cache()
            sys.stdout.flush()
        # we have got shorter embeddings
        # now we need to get the longer embeddings
        longer_embeddings = self.embed_longer_v2(dataset, [-4,-3,-2,-1]) 
        start_time=time.time()
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
        end_time = time.time()  
        # Logging time taken for embedding generation
        print("time taken to embedding list comprehension : ", end_time-start_time)
        return result

if __name__ == "__main__":

    # 1) Toy texts: two shorts and one artificially long
    short_text = "The quick brown fox."
    long_text  = " ".join(["roberta"] * 600)  # ~300 tokens → longer than 128
    texts = [short_text, long_text, short_text]

    model_name = "roberta-large"
    print(f"\n=== Running basic tests on {model_name} ===")

    # 2) Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = TextDataset(texts, tokenizer, max_length=512)

    # Dataset split should be: indices 0 & 2 are short, index 1 is long
    assert ds.short_indices == [0, 2], f"short_indices={ds.short_indices}"
    assert ds.long_indices  == [1],    f"long_indices={ds.long_indices}"
    print("✅ Dataset split OK")

    # Fix the missing shorter_texts list for collate
    ds.shorter_texts = [texts[i] for i in ds.short_indices]

    # 3) Collate/tokenize smoke‐test
    embedder = Embedder(model_name)
    embedder.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder.max_tokens = tokenizer.model_max_length

    batch = embedder.collate_tokenize([short_text, short_text])
    ids  = batch["input_ids"]
    mask = batch["attention_mask"]
    assert ids.shape == mask.shape, "input_ids and attention_mask must match"
    assert ids.shape[0] == 2,      "batch size should be 2"
    assert ids.shape[1] <= 512,    "sequence length must be ≤128"
    print("✅ collate_tokenize OK")

    # 4) Full embed() pipeline
    result = embedder.embed(texts, layers_to_use=[-4, -3, -2, -1])
    embs   = result["embeddings"]
    hidden_size = embedder.model.config.hidden_size

    assert len(embs) == len(texts), f"got {len(embs)} embeddings, expected {len(texts)}"
    for i, vec in enumerate(embs):
        assert isinstance(vec, list), f"embedding[{i}] should be a list"
        assert len(vec) == 4, (
            f"embedding[{i}] length {len(vec)} != 4"
        )
        assert len(vec[0]) == 1024, (
            f"embedding[{i}][0] length {len(vec)} != {hidden_size}"
        )
    print("✅ embed(...) OK on roberta-large")

    print("\n🎉 All tests passed for roberta-large!")
