import os
import csv

folder_path = "coin_dataset"
output_csv = "coin_labels.csv"

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'denomination', 'era', 'side'])


    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            parts = filename.split('_')
            if len(parts) >= 3:
                denomination = parts[0]
                era = parts[1]
                side = parts[2]
                writer.writerow([filename, denomination, era, side])
