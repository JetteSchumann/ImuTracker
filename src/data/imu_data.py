import abc
import copy
import csv
from src.algorithm.quaternion import *
import scipy.io as sio
import itertools
from src.algorithm.data_filter import *

import logging


class ImuTrajectory(object):
    # TODO start_sample or start_time?
    def __init__(self, pos, v, start_sample):
        self._pos = copy.deepcopy(pos)
        self._v = copy.deepcopy(v)
        self._start_sample = start_sample
        self._num_samples = len(self._pos)

        if len(self._pos) != len(self._v):
            logging.getLogger(__name__).WARNING('Different number of pos and v values')

    def __del__(self):
        del self._pos
        del self._v

    def _get_pos(self):
        return self._pos

    def _get_v(self):
        return self._v

    def _get_num_samples(self):
        return self._num_samples

    def _get_start_sample(self):
        return self._start_sample

    pos = property(_get_pos)
    v = property(_get_v)
    num_samples = property(_get_num_samples)
    start_sample = property(_get_start_sample)

class ImuData(object):

    # AXIS CONVENTION: roll -_> x-axis, pitch --> y-axis, yaw --> z-axis

    __metaclass__ = abc.ABCMeta
    gravity_constant = 9.81

    @abc.abstractmethod
    def __init__(self, file):
        self._file = file
        self._typename = 'IMU'
        self._description = ''
        self._sample_rate = 100
        self._num_samples = 0
        self._id = -1
        self._run_offset=0

        # initialize data container for input
        self._acc_local = []
        self._gyro_deg = []
        self._gyro_rad = []
        self._orientation_deg = []
        self._orientation_rad = []
        self._magnetic_field = []
        self._trajectory = None

        # arrays for filtered data
        self._acc_local_filtered = []
        self._gyro_deg_filtered = []
        self._gyro_rad_filtered = []

        # for calculating acc threshold in stationary phase, [start, end]
        self._stationary_phase = []

        # define which information is available
        self._calibrated = False
        self._ahrs = False

        # only used if ahrs is true
        self._quaternions = []
        # roll, pitch, yaw
        self._euler_angles_unwrapped_deg = []

        self._logger = logging.getLogger(__name__)

    def __del__(self):
        del self._acc_local
        del self._gyro_deg
        del self._gyro_rad
        del self._orientation_deg
        del self._orientation_rad
        del self._magnetic_field
        del self._trajectory
        del self._acc_local_filtered
        del self._gyro_deg_filtered
        del self._gyro_rad_filtered
        del self._stationary_phase
        del self._quaternions
        del self._euler_angles_unwrapped_deg


    # for construction of e new ImuData object with same input --> e.g. for comparing raw and filtered data
    # TODO preprocessor returns new ImuData object
    def copy(self):
        return copy.deepcopy(self)

    @abc.abstractmethod
    def _read_data(self):
        """ read input from different csv formats """
        return

    def set_description(self, description):
        self._description = description + ' of ' + self._typename

    def apply_moving_average_acc(self, window_size):

        self._acc_local = DataFilter.moving_average(self._acc_local, window_size)
        self._logger.info('Moving average filter with window size %i to acc_local applied', window_size)

    def _get_description(self):
        return self._description

    def set_trajectory(self, trajectory):
        self._trajectory = trajectory

    def set_stationary_phase(self, phase):
        self._stationary_phase = phase

    def _get_trajectory(self):
        return self._trajectory

    def _get_acc_local(self):
        return self._acc_local

    def _get_gyro_rad(self):
        return self._gyro_rad

    def _get_gyro_deg(self):
        return self._gyro_deg

    def _get_acc_local_filtered(self):
        return self._acc_local_filtered

    def _get_gyro_rad_filtered(self):
        return self._gyro_rad_filtered

    def _get_gyro_deg_filtered(self):
        return self._gyro_deg_filtered

    def _get_magnetic_field(self):
        return self._magnetic_field

    def _get_orientation_rad(self):
        return self._orientation_rad

    def _get_orientation_deg(self):
        return self._orientation_deg

    def _get_index_start(self):
        return self._index_start

    def _get_index_end(self):
        return self._index_end

    def _get_sample_rate(self):
        return self._sample_rate

    def calibrated_available(self):
        return self._calibrated

    def ahrs_available(self):
        return self._ahrs

    def _get_quaternions(self):
        return self._quaternions

    def _get_euler_angles_unwrapped_deg(self):
        return self._euler_angles_unwrapped_deg

    def _get_num_samples(self):
        return self._num_samples

    def _get_stationary_phase(self):
        return self._stationary_phase

    def _get_run_offset(self):
        return self._run_offset

    description = property(_get_description)
    trajectory = property(_get_trajectory)
    acc_local = property(_get_acc_local)
    gyro_rad = property(_get_gyro_rad)
    gyro_deg = property(_get_gyro_deg)
    acc_local_filtered = property(_get_acc_local_filtered)
    gyro_rad_filtered = property(_get_gyro_rad_filtered)
    gyro_deg_filtered = property(_get_gyro_deg_filtered)
    magnetic_field = property(_get_magnetic_field)
    orientation_rad = property(_get_orientation_rad)
    orientation_deg = property(_get_orientation_deg)
    index_start = property(_get_index_start)
    index_end = property(_get_index_end)
    sample_rate = property(_get_sample_rate)
    quaternions = property(_get_quaternions)
    euler_angles_unwrapped_deg = property(_get_euler_angles_unwrapped_deg)
    num_samples = property(_get_num_samples)
    stationary_phase = property(_get_stationary_phase)
    run_offset = property(_get_run_offset)


class SabelImuData(ImuData):
    # stores sensitivity scale factor for full-scale ranges, see sensor specifications
    acc_sensitivity_scale_factors = {2: 16384, 4: 8192, 8: 4096, 16: 2048}
    gyro_sensitivity_scale_factors = {250: 131, 500: 65.5, 1000: 32.8, 2000: 16.4}

    def __init__(self, file, description='', samples_of_interest='', run_offset=0, already_extracted=False, steady_phase=None):

        super().__init__(file)
        self._logger.info('initializing SABEL IMU')
        self._typename = "Sabel IMU"
        self._description = description
        self._alignment = 'NED'

        #if range of interest is given by samples: set start and end indices to read in certain range only
        if isinstance(samples_of_interest, list) and all(isinstance(sample, int) for sample in samples_of_interest):
            #read in data for that range only, data indexing starts at 0
            self._start_run_index_imu = samples_of_interest[0]
            self._end_run_index_imu = samples_of_interest[1]
        #if no range is given read in whole data
        else:
            #TODO check if this is working
            self._logger.info('No IMU range given')
            self._start_run_index_imu = 0
            self._end_run_index_imu = -1

        if self._start_run_index_imu-run_offset < 0:
            self._logger.error('ERROR invalid run_offset. read in can not start at %s ', self._start_run_index_imu-run_offset)

        self._run_offset = run_offset
        self._steady_phase = steady_phase

        if not already_extracted:

            self._read_data()
        else:
            # when reading in extracted data samples for run offset are part of data
            self._start_run_index_imu = self._run_offset
            self._read_extracted_data()

    def __del__(self):
        super().__del__()

    def _read_data(self):
        self._logger.info("read data from " +self._file)

        if 'calibrated' in self._file:
            self._calibrated = True
            self._logger.info('calibrated data available')

        if 'ahrs' in self._file:
            self._ahrs = True
            self._logger.info('ahrs data available')

        # TODO add line with ID read in
        with open(self._file, 'r') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')
            # skip header
            next(data_reader)
            # config data
            line = next(data_reader)
            self._sample_rate = int(line[0])
            self._id = int(line[1])
            self._acc_scale = int(line[4])
            self._gyro_scale = int(line[5])
            self._mag_gain = int(line[6])

            # skip header
            next(data_reader)
            # counter is relative to start_run_index_imu
            counter = 0

            if self._end_run_index_imu is not -1:
                #iterate over certain range only
                rows_start_to_end = itertools.islice(data_reader, self._start_run_index_imu-self._run_offset, self._end_run_index_imu)
            else:
                #iterate over whole range
                rows_start_to_end = data_reader

            # initialzation
            first_row = next(rows_start_to_end)

            self._acc_local = np.array([[float(first_row[3]), float(first_row[4]), float(first_row[5])]])
            self._gyro_deg = np.array([[float(first_row[7]), float(first_row[8]), float(first_row[9])]])
            self._magnetic_field = np.array([[float(first_row[10]), float(first_row[11]), float(first_row[12])]])

            if self._ahrs:
                # init
                self._quaternions = np.array(
                    [Quaternion(float(first_row[13]), float(first_row[14]), float(first_row[15]),
                                float(first_row[16]))])
                self._euler_angles_unwrapped_deg = np.array(
                    [[float(first_row[17]), float(first_row[18]), float(first_row[19])]])
            counter += 1

            for row in rows_start_to_end:
                self._acc_local = np.append(self._acc_local, [[float(row[3]), float(row[4]), float(row[5])]],
                                            axis=0)
                self._gyro_deg = np.append(self._gyro_deg, [[float(row[7]), float(row[8]), float(row[9])]], axis=0)
                self._magnetic_field = np.append(self._magnetic_field, [
                    [float(row[10]), float(row[11]), float(row[12])]], axis=0)
                counter += 1
                if counter % 10000 is 0:
                    self._logger.info('reading in line #'+str(counter))

                if self._ahrs:
                    self._quaternions = np.append(self._quaternions, [
                        Quaternion(float(row[13]), float(row[14]), float(row[15]), float(row[16]))], axis=0)
                    self._euler_angles_unwrapped_deg = np.append(self._euler_angles_unwrapped_deg,
                                                                 [[float(row[17]), float(row[18]), float(row[19])]],
                                                                 axis=0)

        # Calibrate data if necessary
        if not self._calibrated:
            # not calibrated --> sabel calibration, see rCore_AHRS_v3.m
            self._logger.info('sabel imu data not calibrated, read in calibration info')
            mat_file = self._file.replace('.csv', '.mat')
            mat_content = sio.loadmat(mat_file)
            athdata = mat_content['athdata']
            rcore = athdata['rcore']
            struct = rcore[0, 0]
            value = struct['calString']
            calString = value[0][0][0]

            self._acc_local[:, 0] -= calString[0]
            self._acc_local[:, 0] /= calString[3]
            self._acc_local[:, 1] -= calString[1]
            self._acc_local[:, 1] /= calString[4]
            self._acc_local[:, 2] -= calString[2]
            self._acc_local[:, 2] /= calString[5]
            self._acc_local /= SabelImuData.acc_sensitivity_scale_factors[self._acc_scale]

            self._gyro_deg[:, 0] -= calString[6]
            self._gyro_deg[:, 0] /= calString[9]
            self._gyro_deg[:, 1] -= calString[7]
            self._gyro_deg[:, 1] /= calString[10]
            self._gyro_deg[:, 2] -= calString[8]
            self._gyro_deg[:, 2] /= calString[11]
            self._gyro_deg /= SabelImuData.gyro_sensitivity_scale_factors[self._gyro_scale]

            self._magnetic_field[:, 0] += calString[12]
            self._magnetic_field[:, 1] += calString[13]
            self._magnetic_field[:, 2] += calString[14]

            W_inverted = np.array([calString[15:18], calString[18:21], calString[21:24]])
            mag_tmp = np.array([self._magnetic_field[:, 0], self._magnetic_field[:, 1], self._magnetic_field[:, 2]])
            self._magnetic_field = np.array(W_inverted.dot(mag_tmp)).transpose()

        # align magnetometer since it is placed in a different way on the sensor
        tmp = np.copy(self._magnetic_field[:, 1])
        self._magnetic_field[:, 1] = self._magnetic_field[:, 2] * -1
        self._magnetic_field[:, 2] = tmp * -1

        self._acc_local *= ImuData.gravity_constant
        self._gyro_rad = np.array([np.array([math.radians(x) for x in row]) for row in self._gyro_deg])

        self._acc_local_filtered = self._acc_local
        self._gyro_deg_filtered = self._gyro_deg
        self._gyro_rad_filtered = self._gyro_rad
        self._num_samples = len(self._acc_local[:,0])

        if self._end_run_index_imu == -1:
            self._end_run_index_imu=self._num_samples


        self._logger.info('read in finished. #imu_samples: '+str(self._num_samples))

        return

    def _read_extracted_data(self):
        self._logger.info("read data from " +self._file)

        with open(self._file, 'r') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')
            # skip header
            next(data_reader)
            line = next(data_reader)
            self._sample_rate = int(line[1])

            next(data_reader)
            # counter is relative to start_run_index_imu
            counter = 0

            rows_start_to_end = data_reader

            # initialzation
            first_row = next(rows_start_to_end)

            self._acc_local = np.array([[float(first_row[0]), float(first_row[1]), float(first_row[2])]])
            self._gyro_deg = np.array([[float(first_row[3]), float(first_row[4]), float(first_row[5])]])
            self._magnetic_field = np.array([[float(first_row[6]), float(first_row[7]), float(first_row[8])]])

            counter += 1

            for row in rows_start_to_end:
                self._acc_local = np.append(self._acc_local, [[float(row[0]), float(row[1]), float(row[2])]],
                                            axis=0)
                self._gyro_deg = np.append(self._gyro_deg, [[float(row[3]), float(row[4]), float(row[5])]], axis=0)
                self._magnetic_field = np.append(self._magnetic_field, [
                    [float(row[6]), float(row[7]), float(row[8])]], axis=0)
                counter += 1
                if counter % 10000 is 0:
                    self._logger.info('reading in line #'+str(counter))

        self._gyro_rad = np.array([np.array([math.radians(x) for x in row]) for row in self._gyro_deg])

        self._acc_local_filtered = self._acc_local
        self._gyro_deg_filtered = self._gyro_deg
        self._gyro_rad_filtered = self._gyro_rad
        self._num_samples = len(self._acc_local[:,0])

        self._end_run_index_imu = len(self._acc_local[:,0])

        self._logger.info('read in finished. #imu_samples: '+str(self._num_samples))

        return

    def write_data_to_file(self, output_file):

        with open(output_file+'.csv', 'w') as writeFile:

            writeFile.write('# IMU device, sample rate / Hz, number of samples before run starts\n')
            writeFile.write('# SabelLabs, '+str(self._sample_rate)+', '+str(self._run_offset)+'\n')
            writeFile.write('# Format: acc_x / m/s^2,  acc_y / m/s^2, acc_z / m/s^2, gyro_x / °/s, gyro_y / °/s, gyro_z / °/s, mag_x / mGs, mag_y / mGs, mag_z / mGs\n')

            writer = csv.writer(writeFile)

            for i in range(self._num_samples):
                row = [self._acc_local[i,0], self._acc_local[i,1], self._acc_local[i,2], self._gyro_deg[i,0], self._gyro_deg[i,1], self._gyro_deg[i,2], self._magnetic_field[i,0], self._magnetic_field[i,1], self._magnetic_field[i,2]]
                writer.writerow(row)

        writeFile.close()

        self._logger.info('finished. created ' + output_file + ' with #'+str(self._num_samples)+' samples')

    # TODO move to another function?
    def smooth_acc_data(self, window_size):
        self._acc_local_filtered = DataFilter.moving_average(self._acc_local, window_size)
        #self._gyro_deg_filtered = DataFilter.moving_average(self._gyro_deg, window_size)
        #self._gyro_rad_filtered = np.array([np.array([math.radians(x) for x in row]) for row in self._gyro_deg_filtered])
        self._logger.info('acc data smoothed with window size %i', window_size)

    def smooth_gyro_data(self, window_size):
        self._gyro_deg_filtered = DataFilter.moving_average(self._gyro_deg, window_size)
        self._gyro_rad_filtered = np.array([np.array([math.radians(x) for x in row]) for row in self._gyro_deg_filtered])
        self._logger.info('gyro data smoothed with window size %i', window_size)

    def _get_steady_phase(self):
        return self._steady_phase

    steady_phase = property(_get_steady_phase)

# TODO adapt for current state of algorithm
class AndroSensorImuData(ImuData):
    def __init__(self, file, description=''):
        self._logger.info("initialize ANDRO IMU")
        super().__init__(file, description)

        # additional attributes
        self._acc_local_lin = []
        self._typename = "Android Smartphone IMU"
        self._description = description + ' of ' + self._typename

    def _read_data(self):
        # NOTE switch of X and Y data in accordance with axis conventions

        with open(self._file, 'r') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=';')
            # skip header and first line
            next(data_reader)
            # TODO check
            titles = next(data_reader)
            first = 1
            for row in data_reader:

                if first:
                    self._acc_local = np.array([[float(row[0]), float(row[1]), float(row[2])]])
                    self._acc_local_lin = np.array([[float(row[6]), float(row[7]), float(row[8])]])
                    self._gyro_deg = np.array([[float(row[9]), float(row[10]), float(row[11])]])
                    self._gyro_rad = np.array(
                        [[math.radians(float(row[9])), math.radians(float(row[10])), math.radians(float(row[11]))]])
                    self._magnetic_field = np.array([[float(row[12]), float(row[13]), float(row[14])]])
                    self._orientation_deg = np.array([[float(row[16]), float(row[17]), float(row[15])]])
                    self._orientation_rad = np.array(
                        [[math.radians(float(row[16])), math.radians(float(row[17])), math.radians(float(row[15]))]])
                    first = 0

                else:
                    self._acc_local = np.append(self._acc_local, [[float(row[0]), float(row[1]), float(row[2])]],
                                                axis=0)
                    self._acc_local_lin = np.append(self._acc_local_lin,
                                                    [[float(row[6]), float(row[7]), float(row[8])]], axis=0)
                    self._gyro_deg = np.append(self._gyro_deg, [[float(row[9]), float(row[10]), float(row[11])]],
                                               axis=0)
                    self._gyro_rad = np.append(self._gyro_rad, [
                        [math.radians(float(row[9])), math.radians(float(row[10])), math.radians(float(row[11]))]],
                                               axis=0)
                    self._magnetic_field = np.append(self._magnetic_field,
                                                     [[float(row[12]), float(row[13]), float(row[14])]],
                                                     axis=0)
                    self._orientation_deg = np.append(self._orientation_deg,
                                                      [[float(row[16]), float(row[17]), float(row[15])]], axis=0)
                    self._orientation_rad = np.append(self._orientation_rad, [
                        [math.radians(float(row[16])), math.radians(float(row[17])), math.radians(float(row[15]))]],
                                                      axis=0)

        self._id = 1

        self._logger.info('finished read in')

        return
