import csv
import torch

# Read data from CSV file
def read_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

# Convert data to PyTorch tensor
def convert_to_tensor(data):
    # Remove header row if present
    header = data[0]
    data = data[1:]
    # Convert strings to floats or integers
    data = [[float(value) if value.replace('.', '', 1).isdigit() else value for value in row] for row in data]
    return torch.tensor(data)

# Specify the CSV file path
csv_file_path = "disease_info.csv"

# Read data from CSV file
csv_data = read_csv(csv_file_path)

# Convert data to PyTorch tensor
tensor_data = convert_to_tensor(csv_data)

# Specify the file path where you want to save the tensor
pt_file_path = "example_data.pt"

# Save the tensor to a .pt file
torch.save(tensor_data, pt_file_path)

print(f"Tensor saved to {pt_file_path}")
