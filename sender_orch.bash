#!/bin/bash

start_batch=31
end_batch=41
input_base_path="/user/large-scale-embeddings/sampled_data_2019"
output_base_path="/user/large-scale-embeddings/2019_msg_embeddings"
max_retries=3
batch_size1=50000
batch_size2=10000

for batch_num in $(seq $start_batch $end_batch); do
    # input_path="${input_base_path}/2019_sampled_data_11_100_usr_year_week_batch${batch_num}"
    
    # # Create a new output directory in HDFS for each batch
    # output_path="${output_base_path}/batch${batch_num}"
    echo $output_path
    hdfs dfs -mkdir -p "$output_path"
    
    echo "Processing batch $batch_num"
    
    retries=0
    success=0
    
    while [ $retries -lt $max_retries ]; do
        if [ $retries -gt 0 ]; then
            # Add --batch-size argument on retries
            python3 client/sender.py --input_path $input_path --output_path $output_path --batch_size $batch_size1
        elif [ $retries -gt 1]; then
            python3 client/sender.py --input_path $input_path --output_path $output_path --batch_size $batch_size2
        else
            python3 client/sender.py --input_path $input_path --output_path $output_path --batch_size $batch_size1
        fi
        
        if [ $? -eq 0 ]; then
            echo "Batch $batch_num completed successfully."
            success=1
            break
        else
            retries=$((retries+1))
            echo "Error in batch $batch_num, retrying ($retries/$max_retries)..."
            sleep 5  # Optional delay before retrying
        fi
    done
    
    if [ $success -eq 1 ]; then
        status="Success"
    else
        status="Failed"
    fi
    
    if [ $success -eq 0 ]; then
        echo "Batch $batch_num failed after $max_retries retries. Exiting."
        exit 1  # Stop execution if a batch fails after all retries
    fi
done