# server.py
import pyarrow.flight as flight
import pyarrow as pa
from embed import Embedder
import time
import numpy as np
import sys

def setup_logger(log_file):
    """
    Redirects print statements to both terminal and a log file.
    """
    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(log_file, "a")  # Open the log file in append mode

        def write(self, message):
            self.terminal.write(message)  # Print to terminal
            self.log.write(message)  # Write to log file

        def flush(self):
            pass  # This allows for the compatibility of Python 3 print statements

    sys.stdout = Logger()  # Redirect print output to both terminal and file



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
        self.network_time = 0
        self.messages = 0
        self.embedder = Embedder("roberta-large")
        # using last 4 layers
        self.layers_to_use = [-4,-3,-2,-1]

    def getEmbeddings(self, table):

        # function to generate embeddings

        text_data = table.column('message').to_pylist()
        start_embedding_time = time.time()
        result = self.embedder.embed(text_data, self.layers_to_use)
        end_embedding_time = time.time()
        self.embeddings = result['embeddings']

        # Logging Embedding time
        self.embedding_total_time += end_embedding_time - start_embedding_time
        self.tokenization_time += result['tokenization-time']
        self.embedding_generation_time += result['embedding-generation_time']
        self.embedding_aggregation_time += result['embedding-aggregation_time']
        self.data_loading_time += result['data-loading_time']
        if self.messages % 1000000 == 0:
            print("Number of messages : ", self.messages)
            print("Network Time Apollo to Cronus : ", self.network_time)
            print("Embedding Total Time : ",self.embedding_total_time)
            print("Tokenization time : ", self.tokenization_time)
            print("Embedding Generation time : ", self.embedding_generation_time)
            print("Embedding Aggregation time : ", self.embedding_aggregation_time)
            print("Data loading time : ", self.data_loading_time)
        

    def do_put(self, context, descriptor, reader, writer):
        
        # Logging network time to recieve data from apollo
        start_network_time = time.time()
        self.table = reader.read_all()
        end_network_time = time.time()
        self.network_time += (end_network_time - start_network_time)
        self.messages += self.table.num_rows
        self.getEmbeddings(self.table)
    
    def do_get(self, context, ticket):

        # adding embeddings to the table
        updated_table = self.table.append_column('embedding', pa.array(self.embeddings))
        return flight.RecordBatchStream(updated_table)

def start_server():

    log_file = "/home/odeshpande/arrow-communicator/log_file_server.txt"
    setup_logger(log_file)
    server = MyFlightServer(('0.0.0.0', 5111))  # Listen on all interfaces
    print("Starting server on port 5111")
    server.serve()

if __name__ == "__main__":
    start_server()
