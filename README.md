# End-2-End Data Transfer for Deep learning(client with HDFS)

## Description

This repository contains the code for running an Arrow Flight client which can read tweets from Hadoop File system. It can then transfer these tweets from Apollo server to the Cronus(GPU side), then gets back embeddings and stores them on HDFS

## Repository Structure

- `client.py`: This script sets up and runs an Arrow Flight client after preprocessing and batching the data, then makes doGet() and doPut() calls to transfer the tweets and get embeddings back respectively.

## Prerequisites

- Python 3.8 or higher
- Apache Arrow
- Arrow Flight
- NumPy
- Hugging Face
- PyTorch