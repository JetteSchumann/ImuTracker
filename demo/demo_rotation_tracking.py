from src.data.camera_database import *
from src.imu_tracker import *
from src.util.analysis import *
from src.algorithm.distance_tracker import *

import logging
import configparser


def run_rotunda_studies():
    run = '040_N08_h-_R1'
    sensor = 'D7'

    run_data_config = configparser.ConfigParser()
    run_data_config.read(run + '.ini')

    algo_config = configparser.ConfigParser()
    algo_config.read('algo.ini')

    # first and last heel drop of person wearing D7
    global_camera_sync_times_fps = ['00:51:51:14', '01:45:35:17']
    # first and last heel drop of D7
    sync_samples = [133028, 456102]

    fps = int(run_data_config['Info']['Fps'])
    global_run_range_fps = [run_data_config['Info']['RunStart'], run_data_config['Info']['RunEnd']]
    global_camera_sync_times = DataAccessManager.timecode_fps_to_ms(global_camera_sync_times_fps, fps)
    global_run_range = DataAccessManager.timecode_fps_to_ms(global_run_range_fps, fps)

    # TODO supply data files via ped database
    # read in IMU data
    imu_file = '../Input/' + run_data_config[sensor]['File']
    imu_run_offset = int(run_data_config[sensor]['Offset'])

    sample_range_imu = DataAccessManager.get_imu_run_range(imu_file, global_run_range,
                                                           global_camera_sync_times,
                                                           sync_samples)
    imu_data1 = SabelImuData(imu_file, run + sensor, sample_range_imu, imu_run_offset)

    # read in camera data
    petrack_file = run_data_config['Info']['File']
    v_smooth_range = int(algo_config['Algorithm']['VelocitySmoothRange'])
    petrack_database = PeTrackDatabase(petrack_file, v_smooth_range, global_run_range_fps, 25)
    petrack_database.cut_off_negative_frames()

    gt_ID = int(run_data_config[sensor]['PeTrackID'])
    alignment_start = int(run_data_config[sensor]['AlignmentStart'])
    alignment_end = int(run_data_config[sensor]['AlignmentEnd'])

    # TODO remove start_q_calc
    start_q_calc = 1

    imu_tracker = ImuCameraTracker(algo_config, [imu_data1], start_q_calc, petrack_database)
    imu_tracker.set_fusion_data([gt_ID])
    imu_tracker.set_alignment_frames([alignment_start, alignment_end])

    start_frame = int(imu_tracker.ground_truth_trajectory.frames[0])
    end_frame = int(imu_tracker.ground_truth_trajectory.frames[-1])

    time_code_start = DataAccessManager.sample_to_timecode(start_frame, fps)
    time_code_end = DataAccessManager.sample_to_timecode(end_frame, fps)

    imu_tracker.calc_orientation([time_code_start, time_code_end], 0)

    [angle_diff1, gt_heading] = Analysis.calculate_rotation_diff(imu_tracker)

    # TODO move from petrackdatabase to util IO
    petrack_database.write_angles_to_file('rotation_data.txt', imu_tracker, angle_diff1,
                                          'angle diff between movement direction and sensor orientation')


if __name__ == "__main__":  #

    # create root logger, settings apply to loggers in other modules
    logger = logging.getLogger()
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s %(name)s %(funcName)s():%(lineno)i: %(message)s")
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    run_rotunda_studies()
