# HDFS Data Transfer with Apache Arrow Flight

This repository provides a solution to transfer Parquet files stored on HDFS to another server over the network using Apache Arrow Flight. The setup includes server and client components that handle efficient data transfer, embedding generation, and data retrieval.

## Features

- **Apache Arrow Integration**: Transfers data as Arrow tables, utilizing efficient in-memory data representation.
- **RPC Communication**: Uses `PutRPC` to send data from client to server, followed by embedding generation on the server. `GetRPC` enables the client to retrieve processed embeddings after generation.
- **Embedding Generation**: Embedding generation starts on the server after receiving data via `PutRPC`.

## Folder Structure

- **`server/`**  
  Contains code to receive data from the client, read Parquet files as Arrow tables, and perform embedding generation after data transfer.

- **`client/`**  
  Initiates `PutRPC` to transfer data to the server and executes `GetRPC` to retrieve embeddings once generation is complete on the server.

## Quick Start

1. **Client Setup**  
   - Navigate to `client/`.
   - Run the client to send data to the server using `PutRPC`.

2. **Server Setup**  
   - Navigate to `server/`.
   - The server receives the data, performs embedding generation, and makes processed data available for retrieval.

3. **Data Retrieval**  
   - The client executes `GetRPC` to retrieve the generated embeddings from the server.

## Requirements

- Apache Arrow Flight
- HDFS
- Parquet libraries
