import sys
import csv
import os
import scipy.io as sio


def trim_data_csv(file_name, data_type, start, end):
    if not os.path.exists(file_name):
        print(file_name, 'does not exist!')
        sys.exit()

    header_length = 0

    if data_type == 'sabel':
        header_length = 3
        # sampling starts at 1
        sample_counter = 1
    elif data_type == 'motive':
        header_length = 7
        # sampling starts at 0
        sample_counter = 0

    file, extension = os.path.splitext(file_name)
    new_file = file + '_trimmed_' + str(start) + '-' + str(end)
    with open(new_file + extension, 'w', newline='') as output_file:
        data_writer = csv.writer(output_file, delimiter=',')

        with open(file_name, 'r') as input_file:
            data_reader = csv.reader(input_file, delimiter=',')
            # copy header

            for i in range(0, header_length):
                line = next(data_reader)
                data_writer.writerow(line)

            for line in data_reader:
                if sample_counter >= start:
                    data_writer.writerow(line)

                sample_counter += 1

                if sample_counter > end:
                    break

    print('created ' + new_file)

if __name__ == "__main__":
    file_name = sys.argv[1]
    data_type = sys.argv[2]
    start = int(sys.argv[3])
    end = int(sys.argv[4])
    trim_data_csv(file_name, data_type, start, end)
