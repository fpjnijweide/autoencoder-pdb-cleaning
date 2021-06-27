import sys

from src import data_cleaning

if __name__ == "__main__":
    if len(sys.argv)<3:
        filename = "input_data/Dataset - LBP RA small.csv"
        output_filename = "output_data/Dataset - LBP RA small_cleaned.csv"
    else:
        filename = sys.argv[1]
        output_filename = sys.argv[2]

    if len(sys.argv)<4:
        autoencoder_filename = None
    else:
        autoencoder_filename = sys.argv[3]

    data_cleaning.clean(filename,output_filename,autoencoder_filename)