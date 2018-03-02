import numpy as np

def create_data( num_records, num_bits ):
	data = dict()
	data['X'] = np.random.randint(2, size=(num_records, num_bits))
	data['y'] = np.array([list("{0:b}".format(int(''.join([str(x) for x in row]), 2) + 1).zfill(num_bits))[-num_bits:] for row in data['X']], dtype='int64')
	return data
if __name__ == '__main__':
    print(create_data(15,5))
