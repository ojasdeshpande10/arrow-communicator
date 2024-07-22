import pyspark
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
import pyarrow
import pyarrow.flight as flight
import pyarrow as pa
import pyarrow.fs as pafs
import time
import os
import argparse
import random
import re
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

class MyFlightClient(flight.FlightClient):
    
    def __init__(self, server_address='130.245.132.100', port=5111):
        super().__init__(f'grpc://{server_address}:{port}') 
    def sendpyarrow(self, input_table):

        # Approach 1: converting spark dataframe to pandas and then to pyarrow table
        # pandas_df = spark_df.toPandas()   
        # Approach 2: taking input as pyarrow table
        table = input_table
        # descriptor will act as the ID for the data stream being sent
        descriptor = flight.FlightDescriptor.for_path("example_path")
        writer, _ = self.do_put(descriptor, table.schema)
        writer.write_table(table)
        writer.close()
        
    def fetch_data_from_server(self, filesystem, filepath):
        # Create a ticket for the data you want. The content can be anything that your server understands.
        ticket = flight.Ticket('data_request_ticket')
        # Request the data
        reader = self.do_get(ticket)
        # Read the data into a PyArrow Table
        table = reader.read_all()
        tokenization_time = table.column('tokenization_time').to_pylist()[0]
        embedding_time = table.column('embedding_time').to_pylist()[0]
        network_time_1 = table.column('network_time').to_pylist()[0]
        updated_table = table.drop(['tokenization_time'])
        updated_table = updated_table.drop(['network'])
        print("the column names are: ", updated_table.schema.names)
        start_write_hdfs_time = time.time()
        pq.write_table(updated_table,filepath,filesystem=filesystem)
        end_write_hdfs_time =time.time()
        return end_write_hdfs_time-start_write_hdfs_time, tokenization_time, embedding_time, network_time_1



def batch_table(table, batch_size):

    num_rows = table.num_rows

    if num_rows == 0:
        return

    for start_index in range(0,num_rows, batch_size):
        yield table.slice(start_index, min(batch_size, num_rows - start_index))   
    

def main():
    ### Reading parquet files from HDFS into py arrow tables.(without spark)
    start_time = time.time()
    os.environ['ARROW_LIBHDFS_DIR'] = '/home/hlab-admin/hadoop/lib/native/libhdfs.so'
    os.environ['CLASSPATH'] = os.popen('hadoop classpath --glob').read().strip()
    hdfs = pafs.HadoopFileSystem(host='hdfs://apollo-d0',port=9000)
    try:
        read_time_start = time.time()
        dataset = pq.ParquetDataset('/user/large-scale-embeddings/sampled_data_2021/2021_sampled_data_11_100_usr_year_week_batch1',filesystem=hdfs)
        table = dataset.read()
        read_time_end = time.time()
        column_names = table.schema.names
        print("the column names are: ", column_names)
        print("Number of rows: ", table.num_rows)
        print("Number of columns: ", table.num_columns)
    except Exception as e:
        print("error in reading the data: ",e)
    
    # limiting table for testing
    table = table.slice(0, 1000)
    i=0
    network_time = 0
    write_time = 0
    total_embedding_time = 0
    total_tokenization_time = 0
    myclient = MyFlightClient()
    total_messages = 0
    for batch in batch_table(table, 100000):

        print(batch.num_rows)

        total_messages += batch.num_rows
        # Sending data to cronus for generating embedding
        myclient.sendpyarrow(batch)
        filepath = '/user/large-scale-embeddings/demo_embeddings/2021_demo/test_run_22_July'+str(batch)+'.parquet'
        start_get_time = time.time()
        write_time_batch, tokenization_time, embedding_time = myclient.fetch_data_from_server(hdfs, filepath)
        end_get_time = time.time()
        network_time2_batch = (end_get_time-start_get_time)-write_time_batch
        network_time1_batch
        write_time += write_time_batch 
        network_time += network_time_batch
        total_embedding_time += embedding_time
        total_tokenization_time += tokenization_time
    

    print("Read Time : ", read_time_end-read_time_start)
    print("Network Time : ",network_time)
    print("Write Time : ",write_time)
    print("Total Embedding Time : ",total_embedding_time)
    print("Total tokenization Time : ", total_tokenization_time)
    print("Total sort time : ", end_time_sort-start_time_sort)
    end_time = time.time()
    print("total time taken: ", end_time - start_time)

    

if __name__ == "__main__":
    main()
