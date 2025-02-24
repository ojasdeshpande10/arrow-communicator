# server.py
import pyarrow.flight as flight
import pyarrow as pa
from embed import Embedder
import time
import numpy as np
import sys
from dlatk_embed import MessageEmbedder
from embed import Embedder
import threading 
import queue
import os
import argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"




class MyFlightServer(flight.FlightServerBase):

    def __init__(self, location, **kwargs):
        super().__init__(location, **kwargs)
        self.incoming_data_queue = queue.Queue()
        self.embedded_data = {}
        worker = threading.Thread(target=self._embedding_worker, daemon=True)
        worker.start()
        self.embeddings= None  # Instance variable to store embeddings
        self.embeddingcompletion = False
        self.embedding_total_time = 0
        self.network_time_to_cronus = 0
        self.data_loading_time = 0
        self.network_time = 0
        self.messages = 0 

    def _embedding_worker(self):

        while True:
            self.embedder = MessageEmbedder("roberta-large")  
            batch_id, table = self.incoming_data_queue.get()  # block until new data
            # Check for exit condition if needed (not shown)
            print(f"Worker: Embedding for batch_id={batch_id} started...")
            
            message_id = table.column('message_id').to_pylist()
            text_data = table.column('message').to_pylist()
            combined_list = [[mid, text] for mid, text in zip(message_id, text_data)]
            start_embedding_time = time.time()
            print("starting to embed")
            result = self.embedder.get_embeddings(combined_list)
            # result = self.embedder.embed(text_data, [-4, -3, -2, -1])
            end_embedding_time = time.time()
            # self.embeddings = result
            
            # Store the final embedded table in self.embedded_data
            self.embedded_data[batch_id] = result
            print(f"Worker: Embedding for batch_id={batch_id} done!")
         
    def getEmbeddings(self, table):

        # function to generate embeddings
        message_id = table.column('message_id').to_pylist()
        text_data = table.column('message').to_pylist()
        combined_list = [[mid, text] for mid, text in zip(message_id, text_data)]
        start_embedding_time = time.time()
        print("starting to embed")
        result = self.embedder.get_embeddings(combined_list)
        end_embedding_time = time.time()
        self.embeddings = result
        print("time taken to embed : ", end_embedding_time-start_embedding_time)

        
    def do_put(self, context, descriptor, reader, writer):
        
        # Logging network time to recieve data from apollo
        if descriptor.path:
            print(descriptor.path[0].decode())
            batch_id = descriptor.path[0].decode()
        self.table = reader.read_all()
        self.messages += self.table.num_rows
        print(self.table.num_rows)
        self.incoming_data_queue.put((batch_id, self.table))
        print("the Data is put in  the queue")
    
    def do_get(self, context, ticket):

        batch_id = ticket.ticket.decode("utf-8")
        if batch_id not in self.embedded_data:
            # Could raise an error if not ready:
            # raise flight.FlightServerError("Embeddings not ready.")
            # Or return an empty table to indicate "not done yet."
            empty_table = pa.Table.from_arrays([], names=[])
            return flight.RecordBatchStream(empty_table)

        # If ready, return the embedded table
        updated_table = self.table.append_column('embedding', pa.array(self.embedded_data[batch_id]))
        del self.embedded_data[batch_id]
        return flight.RecordBatchStream(updated_table)
        # adding embeddings to the table
        
        return flight.RecordBatchStream()

def start_server(port):
    server = MyFlightServer(('0.0.0.0', port))  # Listen on all interfaces
    print("Starting server on port ", port)
    server.serve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start MyFlightServer with a specific host and port.")
    parser.add_argument("--port", type=int, help="Port to bind the server (default: 5111)")
    args = parser.parse_args()
    start_server(args.port)