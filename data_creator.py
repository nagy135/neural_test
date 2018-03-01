import numpy as np

def create_data( num_records, num_bits ):
    data = np.zeros((num_records, num_bits))
    choices = np.random.choice(num_bits, num_records)

    data[np.arange(num_records), choices] = 1
    return data

if __name__ == '__main__':
    create_data(15,5)
