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
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize the text
        # Suggestion : remove padding can be removed
        inputs = self.tokenizer(text, return_tensors='pt', max_length = self.max_length, truncation = True)
        return {key: val.squeeze(0) for key, val in inputs.items()}


def collate_fn(batch):
    keys =  list(batch[0].keys())       
    batch_dict = {key: [d[key] for d in batch] for key in keys}
    padded_batch = {}
    
    for key in keys:
        if key in ['input_ids', 'attention_mask', 'token_type_ids']:
            padded_batch[key] = torch.nn.utils.rnn.pad_sequence(batch_dict[key], batch_first=True, padding_value=0)
    
    padded_batch['longest_seq'] = padded_batch['input_ids'].shape[1]
    return padded_batch



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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, output_hidden_states=True)
        if torch.cuda.is_available():
            device = torch.device("cuda:0") 
            self.model.to(device)
            self.model = self.model.half()
    def embed_longer(self,texts:List[str], layers_to_use:List[int],  max_chunk_length=510, batch_size=256):
        all_embeddings = []
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id

        # Step 1: Tokenize all messages and split them into chunks
        all_chunks = []
        message_chunk_indices = []

        start_time_tokenizing = time.time()
        for message_idx, message in enumerate(texts):
            tokenized_output = self.tokenizer(message, add_special_tokens=False)

            input_ids = tokenized_output["input_ids"]
            if len(input_ids)>512:
                print(len(input_ids))
            attention_mask = tokenized_output["attention_mask"]

            input_id_chunks = [input_ids[i:i + max_chunk_length] for i in range(0, len(input_ids), max_chunk_length)]
            attention_mask_chunks = [attention_mask[i:i + max_chunk_length] for i in range(0, len(attention_mask), max_chunk_length)]

            chunks = [
                {
                    "input_ids": [cls_token_id] + chunk + [sep_token_id],
                    "attention_mask": [1] + mask + [1],
                    "message_idx": message_idx
                }
                for chunk, mask in zip(input_id_chunks, attention_mask_chunks)
            ]

            all_chunks.extend(chunks)
            message_chunk_indices.extend([message_idx] * len(chunks))
        end_time_tokenizing = time.time()
        # Step 2: Process chunks in batches
        def collate_chunks(chunks):
            max_len = max(len(chunk["input_ids"]) for chunk in chunks)
            input_ids = []
            attention_masks = []
            message_indices = []

            for chunk in chunks:
                ids = chunk["input_ids"]
                mask = chunk["attention_mask"]
                padding_length = max_len - len(ids)

                input_ids.append(ids + [self.tokenizer.pad_token_id] * padding_length)
                attention_masks.append(mask + [0] * padding_length)
                message_indices.append(chunk["message_idx"])

            return {
                "input_ids": torch.tensor(input_ids, device="cuda"),
                "attention_mask": torch.tensor(attention_masks, device="cuda"),
                "message_indices": message_indices
            }

        # Step 3: Batch processing
        chunk_embeddings_dict = {}

        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch = collate_chunks(batch_chunks)

            model_inputs = {k: v for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            message_indices = batch["message_indices"]

            with torch.no_grad():
                outputs = self.model(**model_inputs, output_hidden_states=True)
                hidden_states = torch.stack([outputs.hidden_states[i] for i in layers_to_use], dim=0)  # Shape: (4, batch_size, seq_length, hidden_size)
                attention_mask = batch['attention_mask'].unsqueeze(0).unsqueeze(-1)  # Shape: (1, batch_size, seq_length, 1)
                masked_hidden_states = hidden_states * attention_mask  # Zero-out padded tokens

                # Average over valid tokens
                sum_hidden_states = masked_hidden_states.sum(dim=2)  # Shape: (4, batch_size, hidden_size)
                valid_tokens_count = attention_mask.sum(dim=2)  # Shape: (1, batch_size, 1)
                mean_hidden_states = sum_hidden_states / valid_tokens_count  # Shape: (4, batch_size, hidden_size)

                chunk_embeddings = mean_hidden_states.permute(1, 0, 2)  # Shape: (batch_size, 4, hidden_size)

            # Store embeddings by message index
            for idx, message_idx in enumerate(message_indices):
                if message_idx not in chunk_embeddings_dict:
                    chunk_embeddings_dict[message_idx] = []
                chunk_embeddings_dict[message_idx].append(chunk_embeddings[idx])

        # Step 4: Aggregate chunk embeddings for each message
        for message_idx in range(len(texts)):
            message_chunks = chunk_embeddings_dict[message_idx]
            message_embedding = torch.stack(message_chunks).mean(dim=0)  # Average the chunk embeddings
            all_embeddings.append(message_embedding.cpu().tolist())
        result = {}
        result['embeddings'] = all_embeddings
        return result
    
    def embed(self, texts:List[str], layers_to_use:List[int]):
        result = {}
        # tokenizing the texts 
        start_time_tokenizing = time.time()
        dataset = TextDataset(texts, self.tokenizer)
        end_time_tokenizing = time.time()

        # creating batches in the dataloading with 
        data_loading_time_start = time.time()
        dataloader = DataLoader(dataset, batch_size=512, shuffle=False, collate_fn=collate_fn)
        data_loading_time_end = time.time()

        embedding_generation_time = 0
        embedding_aggregation_time = 0
        all_embeddings=[]
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                # print(batch['longest_seq'], flush=True)
                del batch['longest_seq']
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
                all_embeddings.extend(pooled_embeddings)
                end_embedding_aggregation_time = time.time()

                # Logging time for embedding aggregation
                embedding_aggregation_time += (end_embedding_aggregation_time-start_embedding_aggregation_time)
                torch.cuda.empty_cache()
            

            sys.stdout.flush()
                

        # result['embeddings'] = all_embeddings
        # result['tokenization-time'] = (end_time_tokenizing-start_time_tokenizing)
        # result['data-loading_time'] = (data_loading_time_end-data_loading_time_start)
        # result['embedding-generation_time'] = embedding_generation_time
        # result['embedding-aggregation_time'] = embedding_aggregation_time

        return all_embeddings