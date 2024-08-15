# server.py
import pyarrow.flight as flight
import pyarrow as pa
from embed import Embedder
import time
import numpy as np


class MyFlightServer(flight.FlightServerBase):

    def __init__(self, location, **kwargs):
        super().__init__(location, **kwargs)
        self.embeddings= None  # Instance variable to store embeddings
        self.embeddingcompletion = False
        self.tokenization_time = 0
        self.embedding_generation_time = 0
        self.embedding_aggregation_time = 0
        self.embedding_total_time = 0
        self.network_time_to_cronus = 0
        self.data_loading_time = 0
        self.embedder = Embedder("roberta-large")
        # using last 4 layers
        self.layers_to_use = [-4,-3,-2,-1]

    def getEmbeddings(self, table):

        # function to generate embeddings

        text_data = table.column('message').to_pylist()
        print("**calling embed function**")
        start_embedding_time = time.time()
        result = self.embedder.embed(text_data, self.layers_to_use)
        end_embedding_time = time.time()
        self.embedding_total_time = end_embedding_time - start_embedding_time
        self.embeddings = result['embeddings']
        self.tokenization_time = result['tokenization-time']
        self.embedding_generation_time = result['embedding-generation_time']
        self.embedding_aggregation_time = result['embedding-aggregation_time']
        self.data_loading_time = result['data-loading_time']
        print("**Embedding over for the batch**")

    def do_put(self, context, descriptor, reader, writer):
        
        # Logging network time to recieve data from apollo
        self.table = reader.read_all()
        start_embedding_time = time.time()
        self.getEmbeddings(self.table)
    
    def do_get(self, context, ticket):

        # adding embeddings to the table
        start_update_table_time = time.time()
        updated_table = self.table.append_column('embedding', pa.array(self.embeddings))
        # attaching wall-clock times to table to be sent to apollo
        tokenizing_time = pa.array([self.tokenization_time] * updated_table.num_rows, type=pa.float64())
        embedding_generation_time = pa.array([self.embedding_generation_time] * updated_table.num_rows)
        embedding_aggregation_time = pa.array([self.embedding_aggregation_time] * updated_table.num_rows)
        data_loading_time = pa.array([self.data_loading_time] * updated_table.num_rows)
        embedding_total_time = pa.array([self.embedding_total_time] * updated_table.num_rows)

        updated_table = updated_table.append_column('tokenization_time', pa.array(tokenizing_time))
        updated_table = updated_table.append_column('embedding_generation_time', pa.array(embedding_generation_time))
        updated_table = updated_table.append_column('embedding_aggregation_time', pa.array(embedding_aggregation_time))
        updated_table = updated_table.append_column('data_loading_time', pa.array(data_loading_time))
        updated_table = updated_table.append_column('embedding_total_time', pa.array(embedding_total_time))
        end_update_table_time = time.time()

        update_table_time = pa.array([end_update_table_time-start_update_table_time] * updated_table.num_rows)
        updated_table = updated_table.append_column('update_table_time', pa.array(update_table_time))

        return flight.RecordBatchStream(updated_table)

def start_server():
    server = MyFlightServer(('0.0.0.0', 5111))  # Listen on all interfaces
    print("Starting server on port 5111")
    server.serve()

if __name__ == "__main__":
    start_server()
