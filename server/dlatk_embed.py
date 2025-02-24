import os
import sys

# Dlatk not being built as a package on the dev-transformers branch, hence adding the path to sys.path

sys.path.append(os.path.abspath('/home/odeshpande/dlatk'))

from dlatk.transformer_embs import transformer_embeddings
import numpy as np
from tqdm import tqdm
import torch
import time



class MessageEmbedder:
    def __init__(self, modelName):
        self.messages = []
        self.batch_size = 512
        self.embedding_model = self.create_embedding_model(modelName)
        
    def create_embedding_model(self, modelName):
        print("Creating embedding model")
        cf_embedding_generator = transformer_embeddings(
            modelName=modelName,
            tokenizerName=modelName,
            layersToKeep=[-4, -3, -2, -1],
            aggregations=['mean'],
            layerAggregations=['concatenate'],
            wordAggregations=['mean'],
            maxTokensPerSeg=512,
            batchSize=self.batch_size,
            noContext=False,
            customTableName=None
        )

        return cf_embedding_generator
    def get_embeddings(self, messages):

    # messages is a list of lists, each inner list has two elements, message_id and message
    ############################################################################
        # Batching process
        groupedMessageRows = []
        batch_counter = 0
        message_counter = 0
        no_context = False
        for i in range(0, len(messages), self.batch_size):
            batch_counter += 1
            batch = messages[i:i + self.batch_size]
            # messages are in list of list format, inside list has two elementes msg_id and message
            grouped_batch = [[msg_in[0], msg_in[1]] for msg_in in batch]
            groupedMessageRows.append(["dummy_" + str(batch_counter), grouped_batch])
        all_embeddings = []
        timings = {"prepare_messages": 0, "model_call": 0, "padding":0, "layer_extraction":0, "message_aggregate": 0, "embedding_stacking" : 0}
        for batch in tqdm(groupedMessageRows):
            start_time = time.time()
            tokenIdsDict, (cfId_seq, msgId_seq) = self.embedding_model.prepare_messages([batch], sent_tok_onthefly=True, noContext=no_context)
            end_time = time.time()
            timings["prepare_messages"] += end_time - start_time
            if len(tokenIdsDict["input_ids"]) == 0:
                continue

            # Generate transformer embeddings
            encSelectedLayers = self.embedding_model.generate_transformer_embeddings(tokenIdsDict)
            if encSelectedLayers is None:
                continue
            # Aggregate embeddings at the message level
            start_time = time.time()
            msg_reps, msgIds_new, cfIds_new = self.embedding_model.message_aggregate(encSelectedLayers, msgId_seq, cfId_seq)
            end_time = time.time()
            timings["message_aggregate"] += end_time - start_time
            start_time = time.time()
            msg_reps = np.stack(msg_reps)
            msg_reps = np.squeeze(msg_reps, axis=1)
            pooled_embeddings = np.transpose(msg_reps, (0, 2, 1))
            pooled_embeddings = pooled_embeddings.tolist()
            end_time = time.time()
            timings["embedding_stacking"] += (end_time - start_time)
            all_embeddings.extend(pooled_embeddings)
        
        
        return all_embeddings
        