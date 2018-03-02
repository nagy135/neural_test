import numpy as np

def create_data( num_records, num_bits, my_type="Increment" ):
	data = dict()
	if my_type == "Increment":
		data['X'] = np.random.randint(2, size=(num_records, num_bits))
		data['y'] = np.array([list("{0:b}".format(int(''.join([str(x) for x in row]), 2) + 1).zfill(num_bits))[-num_bits:] for row in data['X']], dtype='float64')
		data['X'] = data['X'].astype('float64')
	if my_type == "Max remains":
		data['X'] = np.random.randint(200, size=(num_records, num_bits))
		data['y'] = list()
		for i,row in enumerate(data['X']):
			index = np.argmax(row)
			item = row[index]
			new_row = np.zeros_like(row)
			new_row[index] = item
			data['y'].append(new_row)
		data['y'] = np.array(data['y'])
	return data
if __name__ == '__main__':
    print(create_data(15,5, "Max remains"))
