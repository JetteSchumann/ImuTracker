import numpy as np

class DataFilter(object):

    @staticmethod
    def moving_average(data, window_size):

        filtered_data = np.zeros((len(data), len(data[0])))

        # floor division
        half_window = window_size//2

        for i in range(len(data)):

            if i < half_window:
                # start phase
                filtering_interval = i

            elif i + half_window >= len(data):
                # end phase
                filtering_interval = len(data) - i - 1

            else:
                filtering_interval = half_window

            sum_xyz = [np.sum(data[i-filtering_interval:i+filtering_interval+1, 0]), np.sum(data[i-filtering_interval:i+filtering_interval+1, 1]), np.sum(data[i-filtering_interval:i+filtering_interval+1, 2])]
            filtered_data[i, :] = [x/(2*filtering_interval+1) for x in sum_xyz]

        return filtered_data

    @staticmethod
    def zero_movement_filter(data, abs_range):

        print('apply zero movement filter for range ', abs_range)

        filtered_data = np.zeros((data.shape[0], data.shape[1]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if -abs_range < data[i,j] < abs_range:
                    filtered_data[i,j] = 0
                    print('set to zero')
                else:
                    filtered_data[i,j] = data[i,j]
        return filtered_data
