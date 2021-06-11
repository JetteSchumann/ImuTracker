import os


def get_TrajData(file_name_in, file_name_out):
    if not os.path.exists(file_name_out):
        # write to file
        file_out = open(file_name_out, "w+")
        with open(file_name_in) as f:
            for line in f:
                if line[0] == '#':
                    # copy header
                    file_out.write(line)
                else:
                    tmp_list = line.split()  # split without an argument -->  split on whitespace
                    # print ('temp_list is: ', tmp_list)
                    # invert x
                    try:
                        x = float(tmp_list[2]) * -(1.0)
                        file_out.write("%s %s %f %s %s \n" % (tmp_list[0], tmp_list[1], x, tmp_list[3], tmp_list[4]))
                    except ValueError as err:
                        print('error, {0}'.format(err))
        file_out.close()

    return


if __name__ == '__main__':
    files_combined = ['040_N08_h-_R1_040_N08_h-_R1_frameshift-3_Combined.txt',
                      '040_N08_h-_R2_040_N08_h-_R2_frameshift-4_Combined.txt',
                      '040_N08_h0_R1_040_N08_h0_R1_frameshift-3_Combined.txt',
                      '040_N08_h0_R2_040_N08_h0_R2_frameshift-3_Combined.txt',
                      '050_N08_h-_R1_050_N08_h-_R1_frameshift-5_Combined.txt',
                      '050_N08_h-_R2_050_N08_h-_R2_frameshift-4_Combined.txt',
                      '050_N08_h0_R1_050_N08_h0_R1_frameshift-4_Combined.txt',
                      '050_N08_h0_R2_050_N08_h0_R2_frameshift-3_Combined.txt',
                      '060_N08_h-_R1_060_N08_h-_R1_frameshift-4_Combined.txt',
                      '060_N08_h-_R2_060_N08_h-_R2_frameshift-3_Combined.txt',
                      '060_N08_h0_R1_060_N08_h0_R1_frameshift-4_Combined.txt',
                      '060_N08_h0_R2_060_N08_h0_R2_frameshift-4_Combined.txt',
                      '070_N08_h-_R1_070_N08_h-_R1_frameshift-4_Combined.txt',
                      '070_N08_h-_R2_070_N08_h-_R2_frameshift-4_Combined.txt',
                      '070_N08_h0_R1_070_N08_h0_R1_frameshift-3_Combined.txt',
                      '070_N08_h0_R2_070_N08_h0_R2_frameshift-5_Combined.txt',
                      '070_N25_h-_R1_070_N25_h-_R1_frameshift0_Combined.txt',
                      '070_N25_h-_R2_070_N25_h-_R2_frameshift-2_Combined.txt',
                      '070_N25_h0_R1_070_N25_h0_R1_frameshift-5_Combined.txt',
                      '070_N25_h0_R2_070_N25_h0_R2_frameshift-2_Combined.txt',
                      '080_N25_h-_R1_080_N25_h-_R1_frameshift-4_Combined.txt',
                      '080_N25_h-_R2_080_N25_h-_R2_frameshift-4_Combined.txt',
                      '080_N25_h0_R1_080_N25_h0_R1_frameshift-5_Combined.txt',
                      '080_N25_h0_R2_080_N25_h0_R2_frameshift-4_Combined.txt',
                      '090_N25_h-_R1_090_N25_h-_R1_frameshift-4_Combined.txt',
                      '090_N25_h-_R2_090_N25_h-_R2_frameshift-5_Combined.txt',
                      '090_N25_h0_R1_090_N25_h0_R1_frameshift-4_Combined.txt',
                      '090_N25_h0_R2_090_N25_h0_R2_frameshift-4_Combined.txt',
                      '100_N25_h-_R1_100_N25_h-_R1_frameshift-3_Combined.txt',
                      '100_N25_h-_R2_100_N25_h-_R2_frameshift-4_Combined.txt',
                      '100_N25_h0_R1_100_N25_h0_R1_frameshift-5_Combined.txt',
                      '100_N25_h0_R2_100_N25_h0_R2_frameshift-2_Combined.txt']

    files_gp = ['040_N08_h-_R1.txt', '040_N08_h-_R2.txt', '040_N08_h0_R1.txt', '040_N08_h0_R2.txt',
                '050_N08_h-_R1.txt', '050_N08_h-_R2.txt', '050_N08_h0_R1.txt', '050_N08_h0_R2.txt',
                '060_N08_h-_R1.txt', '060_N08_h-_R2.txt', '060_N08_h0_R1.txt', '060_N08_h0_R2.txt',
                '070_N08_h-_R1.txt', '070_N08_h-_R2.txt', '070_N08_h0_R1.txt', '070_N08_h0_R2.txt',
                '070_N25_h-_R1.txt', '070_N25_h-_R2.txt', '070_N25_h0_R1.txt', '070_N25_h0_R2.txt',
                '080_N25_h-_R1.txt', '080_N25_h-_R2.txt', '080_N25_h0_R1.txt', '080_N25_h0_R2.txt',
                '090_N25_h-_R1.txt', '090_N25_h-_R2.txt', '090_N25_h0_R1.txt', '090_N25_h0_R2.txt',
                '100_N25_h-_R1.txt', '100_N25_h-_R2.txt', '100_N25_h0_R1.txt', '100_N25_h0_R2.txt']
    input_path = '../Input/RotundaExperiments/PeTrack/Input_combined/'
    output_path = '../Input/RotundaExperiments/PeTrack/'

    for file in files_combined:
        get_TrajData(input_path + file, output_path + file)
