from src import data_cleaning

if __name__ == "__main__":
    # todo get these from args
    filename = "input_data/Dataset - LBP RA small.csv"
    output_filename = "output_data/Dataset - LBP RA small_cleaned.csv"
    autoencoder_filename = None
    data_cleaning.clean(filename,output_filename,autoencoder_filename)