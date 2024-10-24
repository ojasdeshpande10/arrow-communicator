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
from tqdm import tqdm

import sys

class MyFlightClient(flight.FlightClient):
    """
    A client for communicating with an Arrow Flight server.

    This class extends the FlightClient class to provide custom functionality 
    for sending and receiving data between the client and the server using Arrow Flight.
    
    Attributes:
        server_address (str): The IP address of the Flight server.
        port (int): The port number on which the Flight server is listening.
    """ 

    def __init__(self, server_address='130.245.132.100', port=5111):
        """
        Initializes the MyFlightClient with a specified server address and port.

        Args:
            server_address (str): The IP address of the Flight server. Defaults to '130.245.132.100'.
            port (int): The port number on which the Flight server is listening. Defaults to 5111.
        """
        super().__init__(f'grpc://{server_address}:{port}')

    def sendpyarrow(self, input_table):
        """
        Sends a PyArrow table to the server using Arrow Flight.

        Args:
            input_table (pa.Table): The PyArrow Table object to be sent to the server.
        
        Raises:
            flight.FlightError: If the data could not be sent to the server.
        """
        table = input_table
        # descriptor will act as the ID for the data stream being sent
        descriptor = flight.FlightDescriptor.for_path("example_path")
        writer, _ = self.do_put(descriptor, table.schema)
        writer.write_table(table)
        writer.close()
        
    def fetch_data_from_server(self, filesystem, filepath):
        # Create a ticket for the data you want. The content can be anything that your server understands.
        result = {}
        start_network_time = time.time()
        ticket = flight.Ticket('data_request_ticket')
        # Request the data
        reader = self.do_get(ticket)
        # Read the data into a PyArrow Table
        table = reader.read_all()
        updated_table = table.drop(['created_at'])
        end_network_time = time.time()
        
        # Writing the part to the destination folder
        start_write_hdfs_time = time.time()
        pq.write_table(updated_table, filepath, filesystem=filesystem)
        end_write_hdfs_time = time.time()

        result['write-time'] = end_write_hdfs_time - start_write_hdfs_time
        result['network_cronus_to_apollo'] = end_network_time - start_network_time

        return result



def batch_table(table, batch_size):
    """
    Yields batches of rows from the given PyArrow table.

    Args:
        table (pa.Table): The PyArrow Table to be batched.
        batch_size (int): The number of rows per batch.

    Yields:
        pa.Table: A sliced portion of the original table, with up to batch_size rows.
    """
    num_rows = table.num_rows
    if num_rows == 0:
        return

    for start_index in range(0, num_rows, batch_size):
        yield table.slice(start_index, min(batch_size, num_rows - start_index))   

def filter_existing(embedding_path, table, hdfs):
    """
    Filters out rows from the table based on message IDs that already exist in the specified HDFS directory.

    This function checks for existing Parquet files in the specified directory, reads the 'message_id' 
    from each file, and filters out rows from the input table that have already been processed.

    Args:
        embedding_path (str): The HDFS directory path containing the processed Parquet files.
        table (pa.Table): The input PyArrow Table to be filtered.
        hdfs (pafs.HadoopFileSystem): The HDFS filesystem instance used to access the files.

    Returns:
        tuple: A tuple containing the filtered table (pa.Table) and the number of valid files found.
    """
    try:
        file_info = hdfs.get_file_info(pafs.FileSelector(embedding_path))
    except Exception as e:
        print(f"Error fetching file info from {embedding_path}: {e}")
        return None, 0

    num_files = sum(1 for f_info in file_info if f_info.is_file)
    files_to_delete = []  # Keep track of files that need to be deleted

    if num_files > 0:
        all_embedded_message_ids = []  # To store the message IDs from valid files
        

        # Iterate over files in the directory and process each one
        for f_info in file_info:
            if f_info.is_file:
                file_path = f"{embedding_path}/{f_info.base_name}"
                try:
                    # Attempt to read the Parquet file
                    embedded_dataset = pq.ParquetDataset(file_path, filesystem=hdfs)
                    
                    # Process each row group in the dataset
                    for row_group in embedded_dataset.fragments:
                        table_temp = row_group.to_table()
                        embedded_message_ids = table_temp.column('message_id').to_pylist()
                        all_embedded_message_ids.extend(embedded_message_ids)

                except Exception as e:
                    # If there's an error reading the file, log the error and mark for deletion
                    print(f"Error reading file {file_path}: {e}")
                    files_to_delete.append(file_path)

        # Delete the corrupted files
        for file_path in files_to_delete:
            try:
                print(f"Deleting corrupted file: {file_path}")
                hdfs.delete_file(file_path)
            except Exception as delete_error:
                print(f"Error deleting file {file_path}: {delete_error}")

        # Proceed with filtering if valid message IDs were found
        if all_embedded_message_ids:
            embedded_message_ids = pa.array(all_embedded_message_ids)
        else:
            embedded_message_ids = pa.array([])  # No valid files, use an empty array
    else:
        embedded_message_ids = pa.array([])  # No files in the directory

    # Create the mask and filter the table based on the valid message IDs
    mask = pc.is_in(table['message_id'], value_set=embedded_message_ids)
    mask = pc.invert(mask)
    print("The original number of rows:", table.num_rows)
    filtered_table = table.filter(mask)
    print("The filtered table has:", filtered_table.num_rows)

    return filtered_table, num_files - len(files_to_delete)




def main(args):
    """
    The main function for reading, processing, and writing data using PyArrow and Arrow Flight.

    This function sets up logging, reads a Parquet dataset from HDFS, filters out already 
    processed rows, batches the remaining data, and communicates with a Flight server to 
    send and receive processed data. 

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Raises:
        IOError: If there is an issue reading or writing data to HDFS.
    """
    start_time = time.time()
    print("GRPC_MAX_METADATA_SIZE:", os.getenv('GRPC_MAX_METADATA_SIZE'))
    print("GRPC_MAX_SEND_MESSAGE_LENGTH:", os.getenv('GRPC_MAX_SEND_MESSAGE_LENGTH'))
    print("GRPC_MAX_RECEIVE_MESSAGE_LENGTH:", os.getenv('GRPC_MAX_RECEIVE_MESSAGE_LENGTH'))
    
    os.environ['ARROW_LIBHDFS_DIR'] = '/home/hlab-admin/hadoop/lib/native/libhdfs.so'
    os.environ['CLASSPATH'] = os.popen('hadoop classpath --glob').read().strip()
    hdfs = pafs.HadoopFileSystem(host='hdfs://apollo-d0', port=9000)
    print("Batch : ",args.input_path)
    
    try:
        read_time_start = time.time()
        dataset = pq.ParquetDataset(args.input_path, filesystem=hdfs)
        table = dataset.read()
        read_time_end = time.time()
        print("The column names are: ", table.schema.names)
        print("Number of rows: ", table.num_rows)
        print("Number of columns: ", table.num_columns)
    except Exception as e:
        print("Error in reading the data: ", e)
        return
    

    # Filtering already embedded message_ids
    table, file_number = filter_existing(args.output_path, table, hdfs)

    print("the number of rows in filtered table is : ", table.num_rows)
    print("the number of files in the output directory : ", file_number)

    # Limiting table for testing
    if args.limit > 0:
        table = table.slice(0, args.limit)

    i = file_number
    write_time = 0
    network_time2 = 0
    sending_time = 0
    myclient = MyFlightClient(server_address=args.server_address, port=args.port)
    total_messages = 0
    total_batches = (table.num_rows + args.batch_size - 1) // args.batch_size

    for batch in tqdm(batch_table(table, args.batch_size), total=total_batches):
        start_time_batch = time.time()
        total_messages += batch.num_rows 
        # Sending data to the server for generating embedding
        myclient.sendpyarrow(batch)
        filepath = f'{args.output_path}/part{str(i)}.parquet'
        start_get_time = time.time()
        result = myclient.fetch_data_from_server(hdfs, filepath)
        end_get_time = time.time()
        write_time += result['write-time']
        network_time2 += result['network_cronus_to_apollo']
        total_messages += args.batch_size

        if total_messages % 1000000 == 0:
            print("Write Time : ", write_time)
            print("Network Time : ", network_time2)
        i += 1

    print("Read Time : ", read_time_end - read_time_start)
    print("Write Time : ", write_time)
    print("Cronus to Apollo time total : ", network_time2)

    end_time = time.time()
    print("Total time taken: ", end_time - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyArrow and Flight Client Script")
    parser.add_argument("--input_path", type=str, required=True, help="Input HDFS path for the Parquet dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Output HDFS path for the processed Parquet files")
    parser.add_argument("--server_address", type=str, default='130.245.132.182', help="Flight server address")
    parser.add_argument("--port", type=int, default=5111, help="Flight server port")
    parser.add_argument("--batch_size", type=int, default=100000, help="Batch size for processing")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of rows for testing (0 for no limit)")
    args = parser.parse_args()
    
    main(args)
