import os.path
import numpy as np
import sys
import copy
import abc
import src.util.data_access_manager as dam
import src.algorithm.data_filter as df
import logging


class CameraTrajectory(object):

    def __init__(self, identifier, fps, pos, v, frames):
        self._identifier = identifier
        self._fps = fps
        self._pos = copy.deepcopy(pos)
        self._v = copy.deepcopy(v)
        self._frames = copy.deepcopy(frames)

        self._acc = np.array([[0.0, 0.0, 0.0]])
        self._calc_acc()

        logging.getLogger(__name__).info('Identifier %d, start frame: %d, #frames: %d', self._identifier, self._frames[0], len(self._frames))

    def __del__(self):
        del self._pos
        del self._v
        del self._frames
        del self._acc

    def _calc_acc(self):
        for i in range(1, len(self._frames)):
            acc = np.array([(self._v[i] - self._v[i - 1]) / (1 / self._fps)])
            self._acc = np.append(self._acc, acc, axis=0)

    def get_velocity(self, frame):
        index = self.get_index_from_frame(frame)
        return self._v[index]

    def get_position(self, frame):
        index = self.get_index_from_frame(frame)

        return self._pos[index]

    def get_acc(self, frame):
        index = self.get_index_from_frame(frame)
        return self._acc[index]

    def get_index_from_frame(self, frame):
        index = frame - self._frames[0]
        if index < 0 or index > len(self._frames):
            logging.getLogger(__name__).error(
                'INVALID FRAME for trajectory ' + str(self._identifier) + ' , index ' + str(index) + ', len ' + str(
                    len(self._frames)))
            sys.exit()
        return int(index)

    def _get_pos(self):
        return self._pos

    def _get_v(self):
        return self._v

    def _get_acc(self):
        return self._acc

    def _get_frames(self):
        return self._frames

    def _get_identifier(self):
        return self._identifier

    def _get_fps(self):
        return self._fps

    def _get_num_frames(self):
        return len(self._frames)

    identifier = property(_get_identifier)
    pos = property(_get_pos)
    v = property(_get_v)
    acc = property(_get_acc)
    frames = property(_get_frames)
    fps = property(_get_fps)
    num_frames = property(_get_num_frames)


class CameraDatabase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, file, v_average_range=12):
        self._file = file

        # interval for calc average v
        self._v_range = v_average_range
        self._logger = logging.getLogger(__name__)

        if not os.path.exists(self._file):
            self._logger.error(self._file + 'does not exist!')
            sys.exit()

    def _get_fps(self):
        return self._fps

    @abc.abstractmethod
    def get_trajectory(self, identifier):
        return CameraTrajectory()

    fps = property(_get_fps)


class PeTrackDatabase(CameraDatabase):

    # convert SiME data, invert x
    @staticmethod
    def convert_traj(run):

        file_name = '../Input/SiME/PeTrack/' + run + '.txt'

        if not os.path.exists(file_name):
            logging.getLogger(__name__).error('%s does not exist', file_name)
            sys.exit()

        # open, process and close f
        with open(file_name) as f:
            data = np.loadtxt(file_name)

        file_name_converted = '../Input/SiME/PeTrack/' + run + '_converted.txt'

        if not os.path.exists(file_name_converted):
            # write to file
            f_new = open(file_name_converted, "w+")
            with open(file_name) as f:
                for line in f:
                    if line[0] == '#':
                        # copy header
                        f_new.write(line)
                    else:
                        tmp_list = line.split(' ')
                        # invert x
                        x = float(tmp_list[2]) * -(1)
                        f_new.write("%s %s %f %s %s" % (tmp_list[0], tmp_list[1], x, tmp_list[3], tmp_list[4]))
            f_new.close()
        return

    @staticmethod
    def write_angles_to_file(file, imu_tracker, angle_diff, description):

        gt = imu_tracker.ground_truth_trajectory

        # check if file not exists yet --> write header
        if not os.path.exists(file):
            f_new = open(file, "w+")
            f_new.write("# " + description+'\n')
            f_new.write("# id frame diff/°\n")
            f_new.close()

        f_new = open(file, "a+")
        sample_to_frame = imu_tracker.tracking_data[0].imu_to_camera
        camera_frame_start = sample_to_frame[imu_tracker.tracking_data[0].start_tracking_index]
        camera_frame_end = sample_to_frame[imu_tracker.tracking_data[0].end_tracking_index]

        if len(angle_diff) == 2:
            # 2 lists of angle diffs available
            # assumption: angle diff data are available for same frames
            for frame in range(camera_frame_start, camera_frame_end):
                index1 = imu_tracker.tracking_data[0].camera_to_imu[frame] - imu_tracker.tracking_data[0].start_tracking_index
                index2 = imu_tracker.tracking_data[1].camera_to_imu[frame] - imu_tracker.tracking_data[1].start_tracking_index
                f_new.write("%s %s %f %f\n" % (gt.identifier, frame, angle_diff[0][index1], angle_diff[1][index2]))
        else:
            for frame in range(camera_frame_start, camera_frame_end):
                index = imu_tracker.tracking_data[0].camera_to_imu[frame] - imu_tracker.tracking_data[0].start_tracking_index
                f_new.write("%s %s %f \n" % (gt.identifier, frame, angle_diff[index]))
        f_new.close()

    def __init__(self, file, v_average_range=12, run_range=None, moving_average_window=None):
        super().__init__(file, v_average_range)

        # TODO read in
        self._fps = 25

        # TODO if run_range None: init first last frame from data
        self._start_frame = 0
        self._end_frame = int(dam.DataAccessManager.timecode_to_sample(run_range[1],
                                                                       self._fps) - dam.DataAccessManager.timecode_to_sample(run_range[0], self._fps))

        # open, process and close f
        with open(self._file) as f:
            # read in all extracted trajectories
            self._data = np.loadtxt(f)

        self._ped_id = np.unique(self._data[:, 0])
        self._ped_num = len(self._ped_id)

        # fill when requested
        self._ped_dict = {}

        # store data for each person
        self._person_data = {}
        for id in self._ped_id:
            self._person_data[id] = (self._data[self._data[:, 0] == id])

        self._moving_average_window = moving_average_window

        if self._moving_average_window is not None:
            for id in self._ped_id:
                self._person_data[id][:,2:5]=df.DataFilter.moving_average(self._person_data[id][:,2:5], moving_average_window)

        self._logger.info(
            'Initialized petrack camera database for run with ' + str(self._end_frame) + ' frames and ' + str(len(
                self._person_data)) + ' peds')
        self._logger.debug(
            'PETRACK run_range: [' + str(run_range[0]) + ',' + str(run_range[1]) + '] , timecode: [' + str(
                dam.DataAccessManager.timecode_to_sample(
                    run_range[0], self._fps)) + ', ' + str(
                dam.DataAccessManager.timecode_to_sample(run_range[1], self._fps)) + ']')

    # ASSUMPTION: Movement from negative to positive x-axis
    # cuts off trajectory data for x < x_border
    def cut_off_trajectories(self, x_border, start=True):

        for id in self._ped_id:

            num_lines = len(self._person_data[id][:, 2])
            cut_index = 0
            # backwards loop over data
            for i in range(num_lines - 1, 0, -1):
                if self._person_data[id][i, 2] <= x_border:
                    cut_index = i
                    # print('CUT off pedid ', id, ' at ', cut_index)
                    break
            self._person_data[id] = self._person_data[id][cut_index + 1:, :]

    def cut_off_negative_frames(self):
        for id in self._ped_id:

            num_lines = len(self._person_data[id][:, 2])
            cut_index = 0
            # backwards loop over data
            for i in range(num_lines):
                if self._person_data[id][i, 1] >= 0:
                    cut_index = i
                    break
            self._person_data[id] = self._person_data[id][cut_index:, :]

    def _get_num_frames(self):
        return int(self._end_frame - self._start_frame + 1)

    def get_trajectory(self, identifier):
        ped_id = identifier

        if ped_id not in self._ped_dict.keys():
            # create trajectory

            start_frame = int(self._person_data[ped_id][0][1])
            end_frame = int(self._person_data[ped_id][-1][1])
            self._logger.debug(
                'id: ' + str(ped_id) + ', start_frame: ' + str(start_frame) + ', end_frame: ' + str(end_frame))

            pos_data = self._person_data[ped_id][:, 2:5]

            for i in range(len(pos_data)):
                current_frame = int(self._person_data[ped_id][i][1])

                if self._v_range == 0:
                    if i == len(pos_data)-1:
                        #reached end
                        v_mean=[0.0, 0.0, 0.0]
                    else:
                        v_mean = (pos_data[i + 1] - pos_data[i]) / (1 / self._fps)
                else:
                    if i == 0:
                        # very beginning
                        v_mean = (pos_data[i + 1] - pos_data[i]) / (1 / self._fps)
                    elif i == len(pos_data) - 1:
                        # very end
                        v_mean = (pos_data[i - 1] - pos_data[i]) / (1 / self._fps)
                    else:
                        if i < self._v_range:
                            # start phase
                            tmp_v_range = i
                        else:
                            tmp_v_range = self._v_range

                        if i + tmp_v_range > len(pos_data) - 1:
                            # end phase
                            tmp_v_range = len(pos_data) - 1 - i
                        distance = pos_data[i + tmp_v_range] - pos_data[i - tmp_v_range]
                        v_mean = distance / ((tmp_v_range * 2) / self._fps)

                if i == 0:
                    v = np.array(np.array([v_mean]))
                    frames = np.array([current_frame])
                    pos = np.array([pos_data[i]])
                else:
                    v = np.append(v, [v_mean], axis=0)
                    frames = np.append(frames, [current_frame], axis=0)
                    pos = np.append(pos, [pos_data[i]], axis=0)

            print('v size: ', len(v), v[0:2])
            self._ped_dict[ped_id] = CameraTrajectory(ped_id, self._fps, pos, v, frames)



        return self._ped_dict[ped_id]

    num_frames = property(_get_num_frames)


# Length Unit: Meters
# Header Length: 7
class OptiTrackDatabase(CameraDatabase):

    def __init__(self, file, v_average_range=12):
        super().__init__(file, v_average_range)

        # TODO read in
        self._fps = 100
        self._data = np.loadtxt(self._file, delimiter=',', skiprows=7)
        self._frame_offset = int(self._data[0, 0])

        self._logger.info('Initialzed Optitrack database')

    # identifier is marker number: 1 (first), 2 (second), ...
    def get_trajectory(self, identifier):

        col_offset = 2 + 3 * (identifier - 1)
        frames = np.array(self._data[:, 0]) - self._frame_offset
        pos = np.array(self._data[:, col_offset:col_offset + 3])

        # convert right handed y-up system to z-up: swap y and z
        tmp = copy.deepcopy(pos[:, 1])
        pos[:, 1] = -copy.deepcopy(pos[:, 2])
        pos[:, 2] = tmp

        num_samples = len(frames)

        self._logger.debug('id: ' + str(identifier) + ', num frames: ' + str(num_samples))

        for i in range(num_samples):

            if i < self._v_range:
                # start phase
                tmp_v_range = i
            else:
                tmp_v_range = self._v_range

            if i + tmp_v_range > num_samples - 1:
                # end phase
                tmp_v_range = num_samples - 1 - i

            distance = pos[i + tmp_v_range] - pos[i - tmp_v_range]
            v_mean = distance / ((tmp_v_range * 2) / self._fps)

            if i == 0:
                v = np.array(np.array([v_mean]))
            else:
                v = np.append(v, [v_mean], axis=0)

        return CameraTrajectory(identifier, self._fps, pos, v, frames)

    def _get_num_frames(self):
        return len(self._data[:, 0])

    num_frames = property(_get_num_frames)
