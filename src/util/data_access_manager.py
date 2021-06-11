import csv
import numpy as np
import sys
import logging


class DataAccessManager(object):

    @staticmethod
    def _seconds(value, sample_rate):
        if isinstance(value, str):  # value seems to be a timestamp
            _zip_ft = zip((3600, 60, 1, 1 / sample_rate), value.split(':'))
            seconds_sum = sum(f * float(t) for f, t in _zip_ft)
            return seconds_sum
        elif isinstance(value, (int, float, np.int64)):  # frames
            return float(value) / float(sample_rate)
        else:
            return 0

    @staticmethod
    def _timecode(seconds):
        # for x:x:x:frame f=int(round((seconds - int(seconds)) * sample_rate)))
        return '{h:02d}:{m:02d}:{s:02d}.{f:03d}' \
            .format(h=int(seconds / 3600),
                    m=int(seconds / 60 % 60),
                    s=int(seconds % 60),
                    f=int(round((seconds - int(seconds)) * 1000)))

    @staticmethod
    def _samples(seconds, sample_rate):
        return seconds * sample_rate

    @staticmethod
    def timecode_to_sample(timecode, sample_rate, start=None):
        return DataAccessManager._samples(
            DataAccessManager._seconds(timecode, sample_rate) - DataAccessManager._seconds(start, sample_rate),
            sample_rate)

    @staticmethod
    def sample_to_timecode(sample, sample_rate, start=None):
        return DataAccessManager._timecode(
            DataAccessManager._seconds(sample, sample_rate) + DataAccessManager._seconds(start, sample_rate))

    @staticmethod
    def timecode_fps_to_ms(timecode_list, frame_rate):

        new_list = []

        for i in timecode_list:
            separated = i.split(':')
            ms = int(separated[-1]) / frame_rate * 100
            new_list.append(separated[0] + ':' + separated[1] + ':' + separated[2] + ':{t:02d}'.format(t=int(ms)))

        return new_list

    @staticmethod
    # pre condition: data is given for the same starting time/offset
    def get_scaled_indices_imu_camera(imu_data, camera_database):

        imu_run_offset = imu_data.run_offset

        logging.getLogger(__name__).debug('#frames: ' + str(camera_database.num_frames) + ', #samples: ' + str(
            imu_data.num_samples) + ', imu run offset: ' + str(imu_run_offset))
        # scaling over whole data set
        scale_imu_to_camera = camera_database.num_frames / (imu_data.num_samples - imu_run_offset)
        scale_camera_to_imu = (imu_data.num_samples - imu_run_offset) / camera_database.num_frames

        logging.getLogger(__name__).debug(
            'calculated scale imu_camera: ' + str(scale_imu_to_camera) + ', scale camera_imu: ' + str(
                scale_camera_to_imu))

        # calculate indices for fusion range
        imu_to_camera = []
        camera_to_imu = []

        for sample in range(imu_run_offset):
            imu_to_camera.append(-1)

        # TODO check if new indexing is correct with SiME/Rotunda

        prior_frame = -1
        for sample in range(0, imu_data.num_samples - imu_run_offset):
            # round off
            current_frame = int(round(scale_imu_to_camera * sample))
            imu_to_camera.append(current_frame)

            if prior_frame != current_frame:
                camera_to_imu.append(sample + imu_run_offset)
                prior_frame = current_frame
                # if current_frame < 10:
                #    print('frame - sample: ', current_frame, camera_to_imu[-1])

            # if sample < 10:
            #    print('sample - frame: ', sample, imu_to_camera[-1])

        #for frame in range(camera_database.num_frames):
            # round up
        #    camera_to_imu.append(int(round(scale_camera_to_imu * frame + 0.49)) + imu_run_offset)

        #    if frame < 10:
        #        print('frame - sample: ', frame, camera_to_imu[-1])

        for frame in range(camera_database.num_frames):
            sample = camera_to_imu[frame]
            new_frame = imu_to_camera[sample]

            if frame != new_frame:
                logging.getLogger(__name__).warning(
                    'converting indices gone wrong! frame:  %d, camera_to_imu: %d, imu_to_camera: %d', frame, sample,
                    new_frame)

        return [imu_to_camera, camera_to_imu]

    @staticmethod
    # calculates a scale index range for a run defined by timecode_range
    # corresponding syncy are searched for in camera_sync_timestamps
    # imu data are scaled from syncX to syncY. after that the run indices are calulated
    def get_imu_run_range(imu_file, timecode_range, camera_sync_timestamps, manual_syncs=None):

        [imu_sync_data, imu_sample_rate] = DataAccessManager.read_imu_sync_data(imu_file, manual_syncs)

        # if time of interest and camera sync timestamps are given --> scale data within that time range
        # ensure that time range and stamps are a string
        if all(isinstance(timecode, str) for timecode in timecode_range) and all(
                isinstance(timecode, str) for timecode in camera_sync_timestamps):

            if manual_syncs is not None:
                step_size = 1
            else:
                step_size = 2

            # condition: first sync  in camera_sync_times == first sync in imu_sync data, #camera_syncs = #imu_syncs/2, time_of_interest is between two syncs (no more inside)
            if int(len(imu_sync_data) / step_size) is not len(camera_sync_timestamps):
                logging.getLogger(__name__).warning(
                    'WARNING misalignment of syncs. IMU: ' + str(len(imu_sync_data)) + ', CAMERA: ' + str(
                        (len(camera_sync_timestamps))))

            # transform first sync is the offset for all camera data
            abs_camera_start_offset = DataAccessManager.timecode_to_sample(camera_sync_timestamps[0], imu_sample_rate)

            # transform start and end of the run to number of samples, subtract offset
            start_run_sample_from_camera = DataAccessManager.timecode_to_sample(timecode_range[0],
                                                                                imu_sample_rate) - abs_camera_start_offset
            end_run_sample_from_camera = DataAccessManager.timecode_to_sample(timecode_range[1],
                                                                              imu_sample_rate) - abs_camera_start_offset

            logging.getLogger(__name__).info(
                'calculation for IMU samples of interest: ' + str(start_run_sample_from_camera) + ':' + str(
                    end_run_sample_from_camera) +
                ', duration: ' + str(
                    DataAccessManager.sample_to_timecode(end_run_sample_from_camera - start_run_sample_from_camera,
                                                         imu_sample_rate)))

            line_start_sync_camera = -1
            line_end_sync_camera = -1

            # search for camera sync that includes the timecode range
            # loop over list of camera sync timestamps
            for i in range(len(camera_sync_timestamps)):
                # convert current timestamp to sample
                current_sample = DataAccessManager.timecode_to_sample(camera_sync_timestamps[i],
                                                                      imu_sample_rate) - abs_camera_start_offset

                # current sample within run range? --> safe start and end indices of camera timestamps
                if current_sample > start_run_sample_from_camera and line_start_sync_camera == -1:
                    line_start_sync_camera = i - 1
                elif current_sample == start_run_sample_from_camera and line_start_sync_camera == -1:
                    line_start_sync_camera = i

                if current_sample >= end_run_sample_from_camera and line_end_sync_camera == -1:
                    line_end_sync_camera = i

            # get the number of imu samples that should have been recorded
            # camera time is considered as gorund truth
            start_sync_sample_from_camera = DataAccessManager.timecode_to_sample(
                camera_sync_timestamps[line_start_sync_camera],
                imu_sample_rate) - abs_camera_start_offset
            end_sync_sample_from_camera = DataAccessManager.timecode_to_sample(
                camera_sync_timestamps[line_end_sync_camera],
                imu_sample_rate) - abs_camera_start_offset
            num_imu_samples_to_be = end_sync_sample_from_camera - start_sync_sample_from_camera

            logging.getLogger(__name__).debug('camera sync lines: %s - %s, samples: %s - %s, #samples: %s %s ',
                                              line_start_sync_camera,
                                              line_end_sync_camera, start_sync_sample_from_camera,
                                              end_sync_sample_from_camera, num_imu_samples_to_be,
                                              DataAccessManager.sample_to_timecode(num_imu_samples_to_be,
                                                                                   imu_sample_rate))

            # consider: 2 imu sync signals for 1 camera sync
            line_start_sync_imu = -1
            line_end_sync_imu = -1

            imu_start_offset = imu_sync_data[0, 2]

            start_run_sample_imu = start_run_sample_from_camera + imu_start_offset
            end_run_sample_imu = end_run_sample_from_camera + imu_start_offset

            logging.getLogger(__name__).debug(
                'imu samples of interest: %s - %s', start_run_sample_imu, end_run_sample_imu)

            logging.getLogger(__name__).debug(
                'consider step size of %s', step_size)

            for i in range(0, len(imu_sync_data), step_size):

                if imu_sync_data[i, 2] > start_run_sample_imu and line_start_sync_imu == -1:
                    line_start_sync_imu = int(i / step_size) - 1
                elif imu_sync_data[i, 2] == start_run_sample_imu and line_start_sync_imu == -1:
                    line_start_sync_imu = int(i / step_size)

                if imu_sync_data[i, 2] >= end_run_sample_imu and line_end_sync_imu == -1:
                    line_end_sync_imu = int(i / step_size)

            start_sync_sample_imu = imu_sync_data[line_start_sync_imu * step_size, 2]
            end_sync_sample_imu = imu_sync_data[line_end_sync_imu * step_size, 2]
            num_imu_samples = end_sync_sample_imu - start_sync_sample_imu

            logging.getLogger(__name__).debug('FOUND run in imu sync lines ' + str(line_start_sync_imu) + ' : ' + str(
                line_end_sync_imu) + ', samples: ' + str(start_sync_sample_imu) + ' : ' + str(
                end_sync_sample_imu) + ', #samples: ' + str(num_imu_samples) +
                                              str(DataAccessManager.sample_to_timecode(num_imu_samples,
                                                                                       imu_sample_rate)))

            # scale_imu_to_camera = num_imu_samples_to_be / num_imu_samples
            scale_camera_to_imu = num_imu_samples / num_imu_samples_to_be

            logging.getLogger(__name__).debug(
                'num imu samples between syncs: %s, num to be: %s', num_imu_samples, num_imu_samples_to_be)

            start_run_index_imu = int(round(
                start_sync_sample_imu + scale_camera_to_imu * (
                        start_run_sample_from_camera - start_sync_sample_from_camera)))
            end_run_index_imu = int(
                round(
                    end_sync_sample_imu - scale_camera_to_imu * (
                            end_sync_sample_from_camera - end_run_sample_from_camera)))

            logging.getLogger(__name__).info('imu run start and end indices: ' + str(start_run_index_imu) + ':' + str(
                end_run_index_imu) + ', duration: ' + str(
                DataAccessManager.sample_to_timecode(end_run_index_imu - start_run_index_imu, imu_sample_rate)))

        return [start_run_index_imu, end_run_index_imu]

    @staticmethod
    # calculates a scaled index range for a run defined by timecode_range
    # start sync is given only --> manual scale necessary
    def get_imu_run_range_by_start_and_scale(timecode_range, camera_sync_timestamp, imu_sync_sample, imu_sample_rate,
                                             manual_scale_camera_to_imu):

        # if time of interest and camera sync timestamps are given --> scale data within that time range
        # ensure that time range and stamps are a string
        if all(isinstance(timecode, str) for timecode in timecode_range) and isinstance(camera_sync_timestamp, str):
            # transform first sync is the offset for all camera data
            abs_camera_start_offset = DataAccessManager.timecode_to_sample(camera_sync_timestamp, imu_sample_rate)

            # transform start and end of the run to number of samples, subtract offset
            start_run_sample_from_camera = DataAccessManager.timecode_to_sample(timecode_range[0],
                                                                                imu_sample_rate) - abs_camera_start_offset
            end_run_sample_from_camera = DataAccessManager.timecode_to_sample(timecode_range[1],
                                                                              imu_sample_rate) - abs_camera_start_offset

            print('TEST')

            logging.getLogger(__name__).info(
                'calculation of IMU samples of interest: ' + str(start_run_sample_from_camera) + ':' + str(
                    end_run_sample_from_camera) +
                ', duration: ' + str(
                    DataAccessManager.sample_to_timecode(end_run_sample_from_camera - start_run_sample_from_camera,
                                                         imu_sample_rate)))

            # get the number of imu samples that should have been recorded
            # camera time is considered as ground truth
            start_sync_sample_from_camera = DataAccessManager.timecode_to_sample(
                camera_sync_timestamp,
                imu_sample_rate) - abs_camera_start_offset

            start_run_index_imu = int(round(imu_sync_sample + manual_scale_camera_to_imu * (
                        start_run_sample_from_camera - start_sync_sample_from_camera)))

            end_run_index_imu = int(round(imu_sync_sample + manual_scale_camera_to_imu * (
                        end_run_sample_from_camera - start_sync_sample_from_camera)))

            logging.getLogger(__name__).info('imu run start and end indices: ' + str(start_run_index_imu) + ':' + str(
                end_run_index_imu) + ', duration: ' + str(
                DataAccessManager.sample_to_timecode(end_run_index_imu - start_run_index_imu, imu_sample_rate)))
        else:
            logging.getLogger(__name__).error('Invalid format for tracking range or camera sync time stamps. must be timecode.')
            sys.exit(1)

        return [start_run_index_imu, end_run_index_imu]

    @staticmethod
    # sync data format: [TSP, SYNC,  # sample]
    def read_imu_sync_data(imu_file, manual_snycs=None, output_file=None):

        index = 0
        num_syncs = 0
        synchro_info = ''

        with open(imu_file, 'r') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')
            # skip header
            next(data_reader)
            line = next(data_reader)
            sample_rate = int(line[0])
            next(data_reader)
            next(data_reader)

            array_initialized = False

            for row in data_reader:
                tsp_value = float(row[1])
                sync_value = float(row[2])

                if manual_snycs is not None:
                    if index in manual_snycs:
                        if not array_initialized:
                            sync_data = np.array([[tsp_value, sync_value, index]])
                            array_initialized = True
                            num_syncs += 1
                        else:
                            sync_data = np.append(sync_data, [[tsp_value, sync_value, index]], axis=0)
                            num_syncs += 1

                        synchro_info += 'TSP: ' + str("%03d" % tsp_value) + '\t SYNC: ' + str(
                            sync_value) + '\t line: ' + str(
                            "%07d" % index) + '\t time: ' + DataAccessManager.sample_to_timecode(
                            index, sample_rate) + '\n'

                elif sync_value != 0:
                    if not array_initialized:
                        sync_data = np.array([[tsp_value, sync_value, index]])
                        array_initialized = True
                        num_syncs += 1
                    else:
                        sync_data = np.append(sync_data, [[tsp_value, sync_value, index]], axis=0)
                        num_syncs += 1

                    synchro_info += 'TSP: ' + str("%03d" % tsp_value) + '\t SYNC: ' + str(
                        sync_value) + '\t line: ' + str(
                        "%07d" % index) + '\t time: ' + DataAccessManager.sample_to_timecode(
                        index, sample_rate) + '\n'

                index += 1

        logging.getLogger(__name__).debug(synchro_info)
        logging.getLogger(__name__).info('#samples: ' + str(index) + '\t time: ' + DataAccessManager.sample_to_timecode(
            index, sample_rate) + '\t #syncs: ' + str(num_syncs))

        if output_file != None:
            f = open(output_file, 'a+')
            f.write(synchro_info)
            f.write('#samples: ' + str(index) + '\t time: ' + DataAccessManager.sample_to_timecode(
                index, sample_rate) + '\t #syncs: ' + str(num_syncs) + '\n')
            f.close()

        return [sync_data, sample_rate]
