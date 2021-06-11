from src.algorithm.orientation_tracker import *
from src.util.data_access_manager import *
from src.algorithm.distance_tracker import *
from numpy import linalg as LA
import copy
import sys

import logging


class TrackingData(object):

    def __init__(self, imu_data):
        self.imu_data = imu_data
        self.acc_local_lin = []
        self.acc_global = []
        self.acc_global_lin = []

        self.start_tracking_index = 0
        self.end_tracking_index = 0
        self.camera_to_imu = []
        self.imu_to_camera = []
        self.alignment_angle = None

        self.start_q_calc_index = 0
        self.quaternions = np.array([Quaternion()])
        self.quaternions_global = []
        self.euler_angles_rad = []
        self.euler_angles_deg = []
        self.euler_angles_unwrapped_rad = []
        self.euler_angles_unwrapped_deg = []
        self.global_imu_coordinate_axes = []
        # TODO origin necessary?
        self.aligned_global_imu_coordinate_axes = []

        self.acc_adaption_corrected_samples = []
        self.v_orig = []
        self.start_alignment_sample = None
        self.end_alignment_sample = None

        # for analysis purpose only
        self.acc_before_adaption = []
        self.v_before_adaption = []
        self.v_after_adaption = []

    def __del__(self):
        del self.imu_data
        del self.acc_local_lin
        del self.acc_global
        del self.acc_global_lin
        del self.camera_to_imu
        del self.imu_to_camera
        del self.quaternions
        del self.quaternions_global
        del self.euler_angles_rad
        del self.euler_angles_deg
        del self.euler_angles_unwrapped_deg
        del self.euler_angles_unwrapped_rad
        del self.global_imu_coordinate_axes
        del self.acc_adaption_corrected_samples


class ImuTracker(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, algo_config, imu_data, start_q_calc_sample, camera_database=None):

        self._imu_data = imu_data
        self._num_sensors = len(imu_data)
        self._tracking_data = []

        self._orientation_tracker = None
        self._algo_config = algo_config

        rotation_config = self._algo_config['Algorithm']['Rotation']
        if rotation_config == 'Madgwick':
            self._orientation_tracker = MadgwickFilter()
        elif rotation_config == 'Mahony':
            self._orientation_tracker = MahonyFilter(1 / self._imu_data[0].sample_rate)
        else:
            print('Rotation method unknown')
            sys.exit(1)

        self._logger = logging.getLogger(__name__)

        distance_config = self._algo_config['Algorithm']['Distance']
        if distance_config == 'DoubleIntegration':
            self._distance_tracker = DoubleIntegration()
        elif distance_config == 'DoubleIntegrationCorrection':
            self._distance_tracker = DoubleIntegrationCorrection()
        elif distance_config == 'DoubleIntegrationAdaptiveCorrection':
            self._distance_tracker = DoubleIntegrationAdaptiveCorrection()
        elif distance_config == 'MAUKF2D':
            self._distance_tracker = MAUKF2D()
            if self._num_sensors != 2:
                self._logger.error('UKF only implemented for 2 sensors')
                sys.exit(0)
        elif distance_config == 'MAUKF2DCorrection':
            self._distance_tracker = MAUKF2DCorrection()
            if self._num_sensors != 2:
                self._logger.error('UKF only implemented for 2 sensors')
                sys.exit(0)
        elif distance_config == 'MAUKF2DAdaptiveCorrection':
            self._distance_tracker = MAUKF2DAdaptiveCorrection()
            if self._num_sensors != 2:
                self._logger.error('UKF only implemented for 2 sensors')
                sys.exit(0)
        elif distance_config == "None":
            self._distance_tracker = None
            self._logger.info("No distance tracker chosen.")
        else:
            self._logger.error('Distance tracker unknown')
            sys.exit(0)

        for i in range(self._num_sensors):
            self._tracking_data.append(TrackingData(imu_data[i]))
            self._tracking_data[i].start_q_calc_index = start_q_calc_sample - 1

        self._camera_database = camera_database
        # camera database available but fusion data needs to be defined, default False/None
        self._fusion_data_available = False
        self._calculated_trajectory = None

        # TODO several trajectories
        self._ground_truth_ids = []
        self._surrounding_ids = []
        self._ground_truth_trajectory = None
        self._surrounding_trajectory = None
        self._start_pos = [0, 0, 0]
        self._start_velocity = [0, 0, 0]

        self._s = []
        self._v = []

        self._samples_for_convergence = int(self._algo_config['Algorithm']['ConvergenceTime'])
        self._camera_positions_rotated = []

    def __del__(self):
        del self._orientation_tracker
        del self._imu_data
        del self._tracking_data
        del self._ground_truth_trajectory
        del self._surrounding_trajectory
        del self._camera_positions_rotated
        del self._s
        del self._v

    @abc.abstractmethod
    def calc_trajectory(self):
        """Calculates IMU trajectory with choosen IMU tracker.
         returns IMUTrajectory()"""
        return

    @abc.abstractmethod
    def _calc_orientation(self, index):
        return

    # TODO several ids
    def _set_ground_truth_ids(self, id_list):
        self._ground_truth_ids = id_list
        self._ground_truth_trajectory = self._camera_database.get_trajectory(id_list[0])

    # TODO several ids
    def _set_surrounding_ids(self, id_list):
        self._surrounding_ids = id_list
        self._surrounding_ids = self._camera_database.get_trajectory(id_list[0])

    def _get_description(self):
        return 'Tracker for ' + str(self._num_sensors) + ' sensors'

    def _get_v(self):
        return self._v

    def _get_s(self):
        return self._s

    def _get_samples_for_convergence(self):
        return self._samples_for_convergence

    def _get_calculated_trajectory(self):
        return self._calculated_trajectory

    # TODO deep copy??
    def _get_surroundig_trajectory(self):
        return self._surrounding_trajectory

    def _get_ground_truth_trajectory(self):
        return self._ground_truth_trajectory

    def _get_imu_data(self):
        return self._imu_data

    def _get_camera_positions_rotated(self):
        return self._camera_positions_rotated

    def _get_tracking_data(self):
        return self._tracking_data

    def set_alignment_frames(self, frames, sensor_num=0):

        if not self._fusion_data_available:
            self._logger.error('ERROR fusion data needs to be initialized before alignment process')
            sys.exit()

        self._tracking_data[sensor_num].start_alignment_sample = self._tracking_data[sensor_num].camera_to_imu[
            frames[0]]
        self._tracking_data[sensor_num].end_alignment_sample = self._tracking_data[sensor_num].camera_to_imu[frames[1]]
        self._logger.info('SET ALIGNMENT SAMPLE sensor #%d: %d - %d', sensor_num,
                          self._tracking_data[sensor_num].start_alignment_sample,
                          self._tracking_data[sensor_num].end_alignment_sample)

    def set_fusion_data(self, ground_truth_ids, surrounding_ids=None):
        self._fusion_data_available = True

        # TODO handle multiple ids
        self._ground_truth_ids = ground_truth_ids
        self._surrounding_ids = surrounding_ids

        self._ground_truth_trajectory = self._camera_database.get_trajectory(self._ground_truth_ids[0])

        if surrounding_ids is not None:
            self._surrounding_trajectory = self._camera_database.get_trajectory(self._surrounding_ids[0])
            self._v_range = float(self._algo_config['Algorithm']['VelocityDiffSurrounding'])

        for i in range(self._num_sensors):
            [imu_to_camera_tmp, camera_to_imu_tmp] = DataAccessManager.get_scaled_indices_imu_camera(self._imu_data[i],
                                                                                                     self._camera_database)
            self._tracking_data[i].imu_to_camera = imu_to_camera_tmp
            self._tracking_data[i].camera_to_imu = camera_to_imu_tmp

        self._logger.info('FUSION data set')

    description = property(_get_description)
    v = property(_get_v)
    s = property(_get_s)
    tracking_data = property(_get_tracking_data)
    samples_for_convergence = property(_get_samples_for_convergence)
    calculated_trajectory = property(_get_calculated_trajectory)
    surrounding_trajectory = property(_get_surroundig_trajectory)
    ground_truth_trajectory = property(_get_ground_truth_trajectory)
    imu_data = property(_get_imu_data)
    camera_positions_rotated = property(_get_camera_positions_rotated)


class ImuCameraTracker(ImuTracker):
    def __init__(self, algo_config, imu_data, start_q_calc_sample=1, camera_database=None):
        super().__init__(algo_config, imu_data, start_q_calc_sample, camera_database)

        # TODO data_filter as a parameter
        self._filter_data = True

        self._logger.info('Initialized IMU Tracker')

    def __del__(self):
        super().__del__()

    def _set_start_end_tracking_index(self, tracking_range):

        # calculate range for which the tracking algorithm is applied
        # if timecode is given: get corresponding scaled imu samples
        for sensor_num in range(self._num_sensors):

            if all(isinstance(timecode, str) for timecode in tracking_range):

                self._logger.info('Time code given. Calculating sample range.')

                if not self._fusion_data_available:
                    self._logger.error('Tracking range is given as timecode but not camera database is given!')
                    sys.exit()

                # WATCH OUT. round is important
                start_frame = int(
                    round(DataAccessManager.timecode_to_sample(tracking_range[0], self._camera_database.fps)))
                end_frame = int(
                    round(DataAccessManager.timecode_to_sample(tracking_range[1], self._camera_database.fps)))
                print('time code to sample: ',
                      DataAccessManager.timecode_to_sample(tracking_range[0], self._camera_database.fps))
                print('start-end frame:', start_frame, end_frame)
                print('imu_index: ', self._tracking_data[sensor_num].camera_to_imu[
                    start_frame], self._tracking_data[sensor_num].camera_to_imu[
                          end_frame])

                # if run offset is given it is already included in camera_to_imu array
                self._tracking_data[sensor_num].start_tracking_index = self._tracking_data[sensor_num].camera_to_imu[
                    start_frame]
                self._tracking_data[sensor_num].end_tracking_index = self._tracking_data[sensor_num].camera_to_imu[
                    end_frame]

            # if samples are given as range: just copy
            elif all(isinstance(sample, int) for sample in tracking_range):

                self._logger.info('Samples given. Copy.')

                self._tracking_data[sensor_num].start_tracking_index = tracking_range[0]
                self._tracking_data[sensor_num].end_tracking_index = tracking_range[1]

            self._logger.info('sensor #%s , set tracking index range to %s - %s', sensor_num,
                              self._tracking_data[sensor_num].start_tracking_index,
                              self._tracking_data[sensor_num].end_tracking_index)

        if self._num_sensors == 2:
            if self._tracking_data[0].start_tracking_index != self._tracking_data[1].start_tracking_index or \
                    self._tracking_data[0].end_tracking_index != self._tracking_data[1].end_tracking_index:
                self._logger.warning('Different tracking ranges for IMU1 and IMU2: %d-%d, %d-%d',
                                     self._tracking_data[0].start_tracking_index,
                                     self._tracking_data[0].end_tracking_index,
                                     self._tracking_data[1].start_tracking_index,
                                     self._tracking_data[1].end_tracking_index)
                start = max(self._tracking_data[0].start_tracking_index, self._tracking_data[1].start_tracking_index)
                end = min(self._tracking_data[0].end_tracking_index, self._tracking_data[1].end_tracking_index)

                self._tracking_data[0].start_tracking_index = start
                self._tracking_data[0].end_tracking_index = end
                self._tracking_data[1].start_tracking_index = start
                self._tracking_data[1].end_tracking_index = end

                self._logger.warning('set tracking range for both to %d-%d', start, end)

                print('1: ', self._tracking_data[0].start_tracking_index, self._tracking_data[0].end_tracking_index)
                print('2: ', self._tracking_data[1].start_tracking_index, self._tracking_data[1].end_tracking_index)

        self._logger.info('Sample Tracking Range set to %d-%d', self._tracking_data[0].start_tracking_index,
                          self._tracking_data[0].end_tracking_index)

    def _get_adaption_diffs(self):
        return self._adaption_diffs

    def get_rotated_unit_vectors(self):
        return self._angle_x_y_diff

    adaption_diffs = property(_get_adaption_diffs)

    # TODO loop over sensors
    def _get_description(self):
        return 'IMU Tracker for'

    # TODO clear data before calculation
    def calc_orientation(self, tracking_range, sensor_num=0):

        self._set_start_end_tracking_index(tracking_range)

        if self._filter_data:
            self._imu_data[sensor_num].smooth_acc_data(25)

        self._calc_orientation(sensor_num)
        self._calc_euler_angles(sensor_num)

        if self._fusion_data_available and self._tracking_data[sensor_num].start_alignment_sample is not None:
            self._align_coordinate_systems(sensor_num)
        else:
            self._tracking_data[sensor_num].quaternions_global = self._tracking_data[sensor_num].quaternions
            self._tracking_data[sensor_num].aligned_global_imu_coordinate_axes = self._tracking_data[
                sensor_num].global_imu_coordinate_axes

        self._calc_acc_global_rot_acc(sensor_num)

    # TODO clear data before calculation
    def calc_trajectory(self, tracking_range):

        self._logger.info('start tracking')

        self._set_start_end_tracking_index(tracking_range)

        if self._filter_data:
            for i in range(self._num_sensors):
                self._imu_data[i].smooth_acc_data(25)

        # initial values for pos and v if ground truth is given
        # TODO: init values of first sensor are taken
        if self._fusion_data_available:
            self._start_pos = copy.deepcopy(
                self._ground_truth_trajectory.get_position(
                    self._tracking_data[0].imu_to_camera[self._tracking_data[0].start_tracking_index]))
            self._start_velocity = copy.deepcopy(
                self._ground_truth_trajectory.get_velocity(
                    self._tracking_data[0].imu_to_camera[self._tracking_data[0].start_tracking_index]))

        self._logger.debug('start_pos = %s, start_velocity = %s', self._start_pos, self._start_velocity)

        # calc orientation for each sensor
        for sensor_num in range(self._num_sensors):

            self._calc_orientation(sensor_num)

            self._calc_euler_angles(sensor_num)

            if self._fusion_data_available:
                self._align_coordinate_systems(sensor_num)

            else:
                self._tracking_data[sensor_num].quaternions_global = self._tracking_data[sensor_num].quaternions
                self._tracking_data[sensor_num].aligned_global_imu_coordinate_axes = self._tracking_data[
                    sensor_num].global_imu_coordinate_axes

            self._calc_acc_global_rot_acc(sensor_num)

        # CALC POSITION: switch between different distance tracker
        if self._algo_config['Algorithm']['Distance'] == 'DoubleIntegration':
            [self._s, self._v] = self._distance_tracker.calc_positions(self._start_velocity, self._start_pos,
                                                                       self._tracking_data[0])

            print('CREATE TRAJECTORY')
            print('S ', len(self._s), 'V ', len(self._v))
            imu_trajectory = ImuTrajectory(self._s, self._v, self._tracking_data[0].start_tracking_index)
            self._imu_data[0].set_trajectory(imu_trajectory)

        elif self._algo_config['Algorithm']['Distance'] == 'DoubleIntegrationCorrection' or \
                self._algo_config['Algorithm']['Distance'] == 'DoubleIntegrationAdaptiveCorrection':
            [self._s, self._v] = self._distance_tracker.calc_positions(self._start_velocity, self._start_pos,
                                                                       self._tracking_data[0],
                                                                       self._surrounding_trajectory, self._v_range)
            print('CREATE TRAJECTORY')
            print('S ', len(self._s), 'V ', len(self._v))
            imu_trajectory = ImuTrajectory(self._s, self._v, self._tracking_data[0].start_tracking_index)
            self._imu_data[0].set_trajectory(imu_trajectory)

        elif self._algo_config['Algorithm']['Distance'] == 'MAUKF2D':
            self._filtered_values = self._distance_tracker.calc_positions(self._start_velocity, self._start_pos,
                                                                          self._tracking_data)
            self._imu_data[0].set_trajectory(
                ImuTrajectory(self._filtered_values[:, 0:2], self._filtered_values[:, 2:4],
                              self._tracking_data[0].start_tracking_index))
            self._imu_data[1].set_trajectory(
                ImuTrajectory(self._filtered_values[:, 6:8], self._filtered_values[:, 8:10],
                              self._tracking_data[0].start_tracking_index))

        elif self._algo_config['Algorithm']['Distance'] == 'MAUKF2DCorrection' or self._algo_config['Algorithm'][
            'Distance'] == 'MAUKF2DAdaptiveCorrection':
            self._filtered_values = self._distance_tracker.calc_positions(self._start_velocity, self._start_pos,
                                                                          self._tracking_data,
                                                                          self._surrounding_trajectory, self._v_range)
            self._imu_data[0].set_trajectory(
                ImuTrajectory(self._filtered_values[:, 0:2], self._filtered_values[:, 2:4],
                              self._tracking_data[0].start_tracking_index))
            self._imu_data[1].set_trajectory(
                ImuTrajectory(self._filtered_values[:, 6:8], self._filtered_values[:, 8:10],
                              self._tracking_data[0].start_tracking_index))
            print('set trajectories!')
        else:
            self._logger.error('Distance tracking method unknown')
            sys.exit(0)

    # quaternions must be calculated before
    def _calc_euler_angles(self, sensor_num):
        self._tracking_data[sensor_num].euler_angles_rad = np.array(
            [np.array(self._tracking_data[sensor_num].quaternions[0].to_euler3())])

        for i in range(1, len(self._tracking_data[sensor_num].quaternions)):
            self._tracking_data[sensor_num].euler_angles_rad = np.append(
                self._tracking_data[sensor_num].euler_angles_rad,
                np.array([np.array(self._tracking_data[sensor_num].quaternions[i].to_euler3())]), axis=0)

        self._tracking_data[sensor_num].euler_angles_unwrapped_rad = np.array(
            [np.unwrap(self._tracking_data[sensor_num].euler_angles_rad[:, 0]),
             np.unwrap(self._tracking_data[sensor_num].euler_angles_rad[:, 1]),
             np.unwrap(self._tracking_data[sensor_num].euler_angles_rad[:, 2])]).transpose()

        self._tracking_data[sensor_num].euler_angles_deg = np.array(
            [np.array([math.degrees(x) for x in row]) for row in self._tracking_data[sensor_num].euler_angles_rad])
        self._tracking_data[sensor_num].euler_angles_unwrapped_deg = np.array(
            [np.array([math.degrees(x) for x in row]) for row in
             self._tracking_data[sensor_num].euler_angles_unwrapped_rad])

        self._logger.info('Euler angles calculated')

    def _calc_orientation(self, sensor_num):

        counter = 0
        self._logger.info(
            'for sensor #' + str(sensor_num) + ', imu samples: ' + str(
                self._tracking_data[sensor_num].start_q_calc_index) + ' : ' + str(
                self._tracking_data[sensor_num].end_tracking_index))

        q = self._tracking_data[sensor_num].quaternions[0]

        # self._orientation_tracker = MadgwickFilter()

        print('num acc values: ', self._imu_data[sensor_num].acc_local_filtered[:, 0])

        # Calculate q for time steps
        for i in range(self._tracking_data[sensor_num].start_q_calc_index + 1,
                       self._tracking_data[sensor_num].end_tracking_index):
            q_next = self._orientation_tracker.calc_orientation_step(q,
                                                                     self._imu_data[sensor_num].acc_local_filtered[i],
                                                                     self._imu_data[sensor_num].gyro_rad_filtered[i],
                                                                     self._imu_data[sensor_num].magnetic_field[i],
                                                                     counter)

            self._tracking_data[sensor_num].quaternions = np.append(self._tracking_data[sensor_num].quaternions, q_next)
            q = q_next
            counter += 1

        local_imu_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        origin = [0, 0, 0]
        global_axes = [origin]

        for axis in local_imu_axes:
            rotated = self._tracking_data[sensor_num].quaternions[0].rotate_v(axis)
            global_axes.append([rotated[0], rotated[1], rotated[2]])
        self._tracking_data[sensor_num].global_imu_coordinate_axes = np.array([global_axes])

        for i in range(1, len(self._tracking_data[sensor_num].quaternions)):
            global_axes = [origin]
            for axis in local_imu_axes:
                rotated = self._tracking_data[sensor_num].quaternions[i].rotate_v(axis)
                global_axes.append([rotated[0], rotated[1], rotated[2]])

            self._tracking_data[sensor_num].global_imu_coordinate_axes = np.append(
                self._tracking_data[sensor_num].global_imu_coordinate_axes, [global_axes], axis=0)

        self._logger.info('Quaternions calculated. #' + str(
            len(self._tracking_data[sensor_num].quaternions)) + ', #global coords: ' + str(
            len(self._tracking_data[sensor_num].global_imu_coordinate_axes)))

    # sensor is attached with usb up at backrest of wheelchair --> z-axis in direction of movement
    # abs heading of local imu frame = positive z-axis --> get angle in xy-plane
    # compare with direction of movement of gt --> align
    # alignment should be done when gt heading is steady --> TODO search for time range where slope remains the same
    def _align_coordinate_systems(self, sensor_num):

        self._logger.info('alignment calculation started')

        if self._tracking_data[
            sensor_num].start_alignment_sample < self._tracking_data[
            sensor_num].start_q_calc_index + self._samples_for_convergence or \
                self._tracking_data[sensor_num].start_alignment_sample < self._imu_data[sensor_num].run_offset:
            self._logger.error(
                "INVALID alignement range of %d - %d, must start at %d and when camera date ara available after %d",
                self._tracking_data[sensor_num].start_alignment_sample,
                self._tracking_data[sensor_num].end_alignment_sample,
                self._tracking_data[sensor_num].start_q_calc_index + self._samples_for_convergence,
                self._imu_data[sensor_num].run_offset)
            sys.exit()

        camera_pos_start = self._ground_truth_trajectory.get_position(
            self._tracking_data[sensor_num].imu_to_camera[self._tracking_data[sensor_num].start_alignment_sample])[0:2]
        camera_pos_end = self._ground_truth_trajectory.get_position(
            self._tracking_data[sensor_num].imu_to_camera[self._tracking_data[sensor_num].end_alignment_sample])[
                         0:2]

        self._logger.debug('Alignment process within range: %s - %s, camera: %s - %s ',
                           self._tracking_data[sensor_num].start_alignment_sample,
                           self._tracking_data[sensor_num].end_alignment_sample, camera_pos_start, camera_pos_end)

        heading_gt = camera_pos_end - camera_pos_start

        heading_gt_angle = math.degrees(math.atan(heading_gt[1] / heading_gt[0]))

        # -90 < atan(y/x) < 90, convert to 360 deg
        if heading_gt[0] < 0:
            heading_gt_angle += 180
        elif heading_gt[1] < 0:
            heading_gt_angle += 360

        self._logger.debug('heading gt: ' + str(heading_gt) + ', angle: ' + str(heading_gt_angle))

        # get xy coordinates of local imu z-axis
        heading_imu = self._tracking_data[sensor_num].global_imu_coordinate_axes[
                          self._tracking_data[sensor_num].start_alignment_sample - self._tracking_data[
                              sensor_num].start_q_calc_index][3][0:2]

        heading_imu_angle = math.degrees(math.atan(heading_imu[1] / heading_imu[0]))

        # -90 < atan(y/x) < 90, convert to 360 deg
        if heading_imu[0] < 0:
            heading_imu_angle += 180
        elif heading_imu[1] < 0:
            heading_imu_angle += 360

        self._tracking_data[sensor_num].alignment_angle = heading_gt_angle - heading_imu_angle

        self._logger.debug('heading IMU ' + str(heading_imu) + ', angle: ' + str(heading_imu_angle))

        # calc aligned quaterions and aligned global coordinate axis
        q_align = Quaternion.quaternion_from_angle_and_axis(self._tracking_data[sensor_num].alignment_angle, [0, 0, 1])

        self._tracking_data[sensor_num].quaternions_global = []

        for i in range(len(self._tracking_data[sensor_num].quaternions)):
            self._tracking_data[sensor_num].quaternions_global.append(
                q_align * self._tracking_data[sensor_num].quaternions[i])

        # calc aligned local imu frames --> rotate global frame by quaternions_global
        local_imu_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        origin = [0, 0, 0]
        aligned_axes = [origin]

        for axis in local_imu_axes:
            rotated = self._tracking_data[sensor_num].quaternions_global[0].rotate_v(axis)
            aligned_axes.append([rotated[0], rotated[1], rotated[2]])
        self._tracking_data[sensor_num].aligned_global_imu_coordinate_axes = np.array([aligned_axes])

        for q in self._tracking_data[sensor_num].quaternions_global[1:]:
            aligned_axes = [origin]
            for axis in local_imu_axes:
                rotated = q.rotate_v(axis)
                aligned_axes.append([rotated[0], rotated[1], rotated[2]])
            self._tracking_data[sensor_num].aligned_global_imu_coordinate_axes = np.append(
                self._tracking_data[sensor_num].aligned_global_imu_coordinate_axes, [aligned_axes], axis=0)
            # print('global axes: ', self._tracking_data[sensor_num].aligned_global_imu_coordinate_axes[-1], LA.norm(self._tracking_data[sensor_num].aligned_global_imu_coordinate_axes[-1][1,:]))

        self._logger.info('global quaternions and aligned imu coordinate axes calculated. Alignment angle: ' + str(
            self._tracking_data[sensor_num].alignment_angle))

    # pre: quaternions_global are available for each time step --> alginment done, when no fusion data available: quaternions_global = quaternions
    # claculate acc data for availabilty of q --> whole range: [q_start_index:imu_end_tracking_index]
    # calculate global acceleration by rotation of linear acceleration
    # calculate calculate linear global acceleration by subtracting global gravity
    def _calc_acc_global_rot_acc(self, sensor_num):

        # Index offset for accessing imu data
        imu_offset = self._tracking_data[sensor_num].start_q_calc_index

        # Calculate acc_global by rotation and resulting pos
        gravity_global = np.array([0, 0, ImuData.gravity_constant])

        acc_global_tmp = self._tracking_data[sensor_num].quaternions_global[self._samples_for_convergence].rotate_v(
            self._imu_data[sensor_num].acc_local_filtered[imu_offset])
        acc_global_lin_tmp = acc_global_tmp - gravity_global

        acc_local_lin = self._imu_data[sensor_num].acc_local_filtered[imu_offset] - \
                        self._tracking_data[sensor_num].quaternions_global[
                            self._samples_for_convergence].get_inverse().rotate_v(gravity_global)

        self._tracking_data[sensor_num].acc_global = np.array([acc_global_tmp])
        self._tracking_data[sensor_num].acc_global_lin = np.array([acc_global_lin_tmp])
        self._tracking_data[sensor_num].acc_local_lin = np.array([acc_local_lin])

        self._acc_global_xy_angle = []
        acc_prior_normalized = acc_global_lin_tmp / LA.norm(acc_global_lin_tmp)

        for i in range(1, len(self._tracking_data[sensor_num].quaternions)):

            if i < self._samples_for_convergence:
                # take first stable quaternion. must be the same since sensor is not moving
                quat = self._tracking_data[sensor_num].quaternions_global[self._samples_for_convergence]
            else:
                quat = self._tracking_data[sensor_num].quaternions_global[i]

            acc_global_tmp = quat.rotate_v(self._imu_data[sensor_num].acc_local_filtered[imu_offset + i])
            # print('acc_global_tmp: ', acc_global_tmp)

            acc_global_lin_tmp = acc_global_tmp - gravity_global
            # acc_global_lin_tmp[2] = 0

            # local gravity
            local_gravity = quat.get_inverse().rotate_v(gravity_global)
            acc_local_lin_tmp = self._imu_data[sensor_num].acc_local_filtered[imu_offset + i] - local_gravity
            # print('local gravity: ', local_gravity, LA.norm(local_gravity))

            # acc_local_lin_tmp[0:2]=0

            acc_global_new = quat.rotate_v(acc_local_lin_tmp)
            # acc_global_new[2] = 0
            # print('PRIOR: ', acc_global_lin_tmp,', NEW: ', acc_global_new, ', diff: ', LA.norm(acc_global_lin_tmp-acc_global_new ))
            acc_global_lin_tmp = acc_global_new

            # print('acc_local: ', self._imu_data[sensor_num].acc_local_filtered[imu_offset])
            # print('acc_local_lin: ', acc_local_lin, ', gravity_local: ', local_gravity)

            current_acc_normalized = acc_global_lin_tmp / LA.norm(acc_global_lin_tmp)
            angle_xy = math.degrees(
                math.acos(acc_prior_normalized.dot(current_acc_normalized)))
            self._acc_global_xy_angle.append(angle_xy)

            acc_prior_normalized = current_acc_normalized

            self._tracking_data[sensor_num].acc_global = np.append(self._tracking_data[sensor_num].acc_global,
                                                                   [acc_global_tmp], axis=0)
            self._tracking_data[sensor_num].acc_global_lin = np.append(self._tracking_data[sensor_num].acc_global_lin,
                                                                       [acc_global_lin_tmp], axis=0)
            self._tracking_data[sensor_num].acc_local_lin = np.append(self._tracking_data[sensor_num].acc_local_lin,
                                                                      [acc_local_lin_tmp], axis=0)

        # noise when steady
        if self._imu_data[sensor_num].steady_phase is not None:
            start = self._imu_data[sensor_num].steady_phase[0]
            end = self._imu_data[sensor_num].steady_phase[1]
            if start < self._tracking_data[sensor_num].start_q_calc_index + self._samples_for_convergence:
                self._logger.warning('calculating bias in possibly not stable range: %i - %i ', start, end)

            bias_x_loc = np.mean(self._tracking_data[sensor_num].acc_local_lin[start:end, 0])
            bias_y_loc = np.mean(self._tracking_data[sensor_num].acc_local_lin[start:end, 1])
            bias_z_loc = np.mean(self._tracking_data[sensor_num].acc_local_lin[start:end, 2])

            bias_x_glob = np.mean(self._tracking_data[sensor_num].acc_global_lin[start:end, 0])
            bias_y_glob = np.mean(self._tracking_data[sensor_num].acc_global_lin[start:end, 1])
            bias_z_glob = np.mean(self._tracking_data[sensor_num].acc_global_lin[start:end, 2])

            self._tracking_data[sensor_num].acc_local_lin[:, 0] -= bias_x_loc
            self._tracking_data[sensor_num].acc_local_lin[:, 1] -= bias_y_loc
            self._tracking_data[sensor_num].acc_local_lin[:, 2] -= bias_z_loc

            self._tracking_data[sensor_num].acc_global_lin[:, 0] -= bias_x_glob
            self._tracking_data[sensor_num].acc_global_lin[:, 1] -= bias_y_glob
            self._tracking_data[sensor_num].acc_global_lin[:, 2] -= bias_z_glob

            print('BIAS loc: ', bias_x_loc, bias_y_loc, bias_z_loc)
            print('BIAS glob: ', bias_x_glob, bias_y_glob, bias_z_glob)

        self._logger.debug('acc_global_lin = acc_global_tmp - gravity_global at index %s',
                           self._samples_for_convergence)
        self._logger.debug(' %s = %s - %s ',
                           self._tracking_data[sensor_num].acc_global_lin[self._samples_for_convergence],
                           self._tracking_data[sensor_num].acc_global[self._samples_for_convergence], gravity_global)

        print('# acc_global: ', len(self._tracking_data[sensor_num].acc_global_lin))
        # print('angle xy ', self._acc_global_xy_angle)

    def _calc_acc_heading_adaption(self, sensor_num=0):

        for imu_index in range(len(self._tracking_data[sensor_num].aligned_global_imu_coordinate_axes)):

            # estimated heading = calculated global aligned z-axis of the sensor
            imu_z_vector = self._tracking_data[sensor_num].aligned_global_imu_coordinate_axes[imu_index, 3][0:2]
            imu_heading = math.degrees(math.atan(imu_z_vector[1] / imu_z_vector[0]))

            if imu_z_vector[0] < 0:
                imu_heading += 180
            elif imu_z_vector[1] < 0:
                imu_heading += 360

            # acc heading = x,y of global lin acc
            acc_vector = self._tracking_data[sensor_num].acc_global_lin[imu_index][0:2]
            acc_heading = math.degrees(math.atan(acc_vector[1] / acc_vector[0]))

            if acc_vector[0] < 0:
                acc_heading += 180
            elif acc_vector[1] < 0:
                acc_heading += 360

            # calc diff as smallest angle between both
            diff = imu_heading - acc_heading

            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360

            imu_z_vector_normalized = imu_z_vector / LA.norm(imu_z_vector)

            if abs(diff) <= 90:
                # forwards movement
                new_acc = imu_z_vector_normalized * LA.norm(acc_vector)
            else:
                # backwards movement
                new_acc = imu_z_vector_normalized * -LA.norm(acc_vector)

            self.tracking_data[sensor_num].acc_global_lin[imu_index] = [new_acc[0], new_acc[1], 0]

            print('#', imu_index, ', imu heading: ', imu_heading, ', acc heading: ', acc_heading, ', diff: ', abs(diff))
