import os
import csv

def save_data(filename, labels):
    for label in labels:
        folder_path = '../data/{}'.format(label)
        images = os.listdir(folder_path)

        with open(filename, 'a') as csvfile:  
            csvwriter = csv.writer(csvfile)
            
            for image in images:
                row = [image, label]
                csvwriter.writerow(row)

if __name__ == '__main__':
    labels = os.listdir('../data')
    filename = '../data-1.csv'
    save_data(filename, labels)
    

