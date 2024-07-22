# server.py
import pyarrow.flight as flight
import pyarrow as pa
import pandas as pd
from embed import Embedder
import time
import numpy as np

class MyFlightServer(flight.FlightServerBase):

    def __init__(self, location, **kwargs):
        super().__init__(location, **kwargs)
        self.embeddings= None  # Instance variable to store embeddings
        self.embeddingcompletion = False
        self.tokenization_time = 0
        self.embedding_time = 0
        self.network_time_to_cronus = 0

    def getEmbeddings(self, table):
        text_data = table.column('message').to_pylist()
        # put model in init
        embedder = Embedder("roberta-base")
        self.embeddings, tokenization_time, embedding_time = embedder.embed(text_data)
        self.tokenization_time = tokenization_time
        self.embedding_time = embedding_time    
        print(self.embeddings[0].shape)
    def do_put(self, context, descriptor, reader, writer):
        
        # Measuring network time to recieve data from apollo
        start_network_time = time.time()
        self.table = reader.read_all()
        end_network_time = time.time()
        self.network_time_to_cronus = end_network_time - start_network_time
        self.getEmbeddings(self.table)
    
    def do_get(self, context, ticket):

        # adding embeddings to the table
        updated_table = self.table.append_column('embedding', pa.array(self.embeddings))

        # attaching wall-clock times to table to be sent to apollo
        tokenizing_time = pa.array([self.tokenization_time] * result_table.num_rows, type=pa.float64())
        embedding_time = pa.array([self.embedding_time] * result_table.num_rows)
        network_time = pa.array([self.network_time_to_cronus] * result_table.num_rows)

        updated_table = result_table.append_column('tokenization_time', pa.array(tokenizing_time))
        updated_table = updated_table.append_column('embedding_time', pa.array(embedding_time))
        updated_table = updated_table.append_column('network_time', pa.array(network_time))


        return flight.RecordBatchStream(updated_table)

def start_server():
    server = MyFlightServer(('0.0.0.0', 5111))  # Listen on all interfaces
    print("Starting server on port 5111")
    server.serve()

if __name__ == "__main__":
    start_server()
