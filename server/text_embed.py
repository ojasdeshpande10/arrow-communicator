from embed import Embedder
import csv
import numpy as np

# Initialize a dictionary to store column data
columns = {}

# Read the CSV file
file_path = '/home/odeshpande/dep_LBA.csv'  # Replace with the path to your CSV file
with open(file_path, mode='r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Extract the header row
    # Initialize lists for each column in the dictionary
    for column_name in header:
        columns[column_name] = []
    # Populate the lists with data from each row
    for row in reader:
        for col_name, value in zip(header, row):
            columns[col_name].append(value)

print(columns['message'])
print(len(columns['message']))



embedder = Embedder("roberta-large")
layers_to_use = [-4,-3,-2,-1]
result = embedder.embed(columns['message'], layers_to_use)


feat_table = {}
data = []
data.append(['id', 'group_id', 'feat', 'value'])
print(len(columns['message_id']))

for i, message_id in enumerate(columns['message_id']):
    embedding_np = np.array(result['embeddings'][i][-2])
    # if columns['message'][i] != "NA":
    for i in range(len(embedding_np)):
        data.append([message_id, message_id,  f"{i}me", embedding_np[i]])

print(len(data))

csv_file_path = '/home/odeshpande/dep_feat_table.csv'

with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(data)




