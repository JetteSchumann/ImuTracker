import logging
import os
import sys

from src.algorithm.quaternion import *

class Analysis(object):

    @staticmethod
    def calc_diff_in_velocity(wheelchair_traj, pusher_traj, config_name, file_name):

        offset=wheelchair_traj.fps
        start=max(pusher_traj.frames[0], wheelchair_traj.frames[0])+offset
        end=min(pusher_traj.frames[-1], wheelchair_traj.frames[-1])-offset

        v_diff=[]
        v_diff_norm=[]
        for frame in range(start, end):
            v_diff.append(LA.norm(wheelchair_traj.get_velocity(frame) - pusher_traj.get_velocity(frame)))
            v_diff_norm.append(LA.norm(wheelchair_traj.get_velocity(frame)) - LA.norm(pusher_traj.get_velocity(frame)))

        if file_name is not None:
            f = open(file_name, 'a+')
            f.write('#' + config_name + '\n')
            #format: mean(norm(wheeler-pusher)) max(norm_diff) min(norm_diff) mean(norm(wheeler)-norm(pusher)) max(diff_norm) min(diff_norm)
            f.write(
                '{:>3.2f} {:>3.2f} {:>3.2f} {:>3.2f} {:>3.2f} {:>3.2f}\n'.format(np.mean(v_diff), np.max(v_diff), np.min(v_diff), np.mean(v_diff_norm), np.max(v_diff_norm), np.min(v_diff_norm)))
            f.close()

    @staticmethod
    def validate_diff_in_velocity(file_name):

        if not os.path.exists(file_name):
            logging.getLogger(__name__).error('%s does not exist', file_name)
            sys.exit()

            # open, process and close f
        with open(file_name):
            data = np.loadtxt(file_name)

        logging.getLogger(__name__).info('mean_mean: %f, mean_mean_abs: %f, mean_max: %f, mean_min %f, mean_norm_mean: %f, mean_norm_max: %f, mean_norm_min: %f',
                                         np.mean(data[:, 0]), np.mean(data[:, 1]), np.mean(data[:, 2]),
                                         np.mean(data[:, 3]), np.mean(data[:, 4]), np.mean(data[:, 5]), np.mean(data[:, 6]))

    @staticmethod
    # compare calculated heading angle (aligned global y-axis) with angle of tangent to horizontal body axes
    def validate_rotation_optitrack(imu_tracker, gt_left, gt_right, file=None, sensor_num=0):

        angle_diff = np.zeros(imu_tracker.tracking_data[sensor_num].end_tracking_index - imu_tracker.tracking_data[
            sensor_num].start_tracking_index)
        estimated_heading = np.zeros(
            imu_tracker.tracking_data[sensor_num].end_tracking_index - imu_tracker.tracking_data[
                sensor_num].start_tracking_index)
        gt_heading = np.zeros(imu_tracker.tracking_data[sensor_num].end_tracking_index - imu_tracker.tracking_data[
            sensor_num].start_tracking_index)

        for imu_index in range(imu_tracker.tracking_data[sensor_num].start_tracking_index,
                               imu_tracker.tracking_data[sensor_num].end_tracking_index):
            camera_frame = imu_tracker.tracking_data[sensor_num].imu_to_camera[imu_index]

            body_axis = gt_left.get_position(camera_frame) - gt_right.get_position(camera_frame)

            gt_heading_body_axis = math.degrees(math.atan(body_axis[1] / body_axis[0]))

            # -90 < atan(y/x) < 90, convert to 360 deg
            if body_axis[0] < 0:
                gt_heading_body_axis += 180
            elif body_axis[1] < 0:
                gt_heading_body_axis += 360

            # print('body_axis: ', body_axis, ', angle: ', heading_angle)

            # estimated heading = calculated global aligned y-axis of the sensor
            imu_global_y_x = \
                imu_tracker.tracking_data[sensor_num].aligned_global_imu_coordinate_axes[
                    imu_index - imu_tracker.tracking_data[sensor_num].start_q_calc_index, 2][
                    0]
            imu_global_y_y = \
                imu_tracker.tracking_data[sensor_num].aligned_global_imu_coordinate_axes[
                    imu_index - imu_tracker.tracking_data[sensor_num].start_q_calc_index, 2][
                    1]
            estimated_heading_body_axis = math.degrees(math.atan(imu_global_y_y / imu_global_y_x))

            if imu_global_y_x < 0:
                estimated_heading_body_axis += 180
            elif imu_global_y_y < 0:
                estimated_heading_body_axis += 360

            # calc gt heading, vertical to body axes, to campare with local imu z-axes
            gt_heading_angle = gt_heading_body_axis - 90
            if gt_heading_angle < 0:
                gt_heading_angle = 360 + gt_heading_angle

            # calc estimated heading
            estimated_heading_angle = estimated_heading_body_axis - 90
            if estimated_heading_angle < 0:
                estimated_heading_angle = 360 + estimated_heading_angle

            estimated_heading[
                imu_index - imu_tracker.tracking_data[sensor_num].start_tracking_index] = estimated_heading_angle

            gt_heading[imu_index - imu_tracker.tracking_data[sensor_num].start_tracking_index] = gt_heading_angle

            diff = estimated_heading_body_axis - gt_heading_body_axis

            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360

            angle_diff[imu_index - imu_tracker.tracking_data[sensor_num].start_tracking_index] = diff

        mean = np.mean(angle_diff)
        mean_abs = np.mean(abs(angle_diff))
        max = np.max(angle_diff)
        min = np.min(angle_diff)
        max_abs = np.max(abs(angle_diff))
        logging.getLogger(__name__).info('mean error: %f, mean_abs: %f, max: %f, min: %f, max_abs: %f', mean, mean_abs,
                                         max, min, max_abs)

        if file is not None:
            config = imu_tracker.imu_data[sensor_num].description

            f = open(file, 'a+')
            f.write('{:10} {:>5.2f} {:>5.2f} {:>5.2f} {:>5.2f} {:>5.2f}\n'.format(config, mean, mean_abs, max, min,
                                                                                  max_abs))
            f.close()

        return [angle_diff, estimated_heading, gt_heading]

    @staticmethod
    # compare calculated heading angle (aligned global y-axis) with angle of tangent to horizontal body axes
    def validate_rotation_optitrack_without_alignment(imu_tracker, gt_left, gt_right, file=None, sensor_num=0):

        angle_diff = np.zeros(imu_tracker.tracking_data[sensor_num].end_tracking_index - imu_tracker.tracking_data[
            sensor_num].start_tracking_index)
        estimated_heading = np.zeros(
            imu_tracker.tracking_data[sensor_num].end_tracking_index - imu_tracker.tracking_data[
                sensor_num].start_tracking_index)
        gt_heading = np.zeros(
            imu_tracker.tracking_data[sensor_num].end_tracking_index - imu_tracker.tracking_data[
            sensor_num].start_tracking_index)

        offset = 0
        priot_gt_heading = 0
        priot_estimated_heading = 0

        for imu_index in range(imu_tracker.tracking_data[sensor_num].start_tracking_index,
                               imu_tracker.tracking_data[sensor_num].end_tracking_index):
            camera_frame = imu_tracker.tracking_data[sensor_num].imu_to_camera[imu_index]

            body_axis = gt_left.get_position(camera_frame) - gt_right.get_position(camera_frame)

            gt_heading_body_axis = math.degrees(math.atan(body_axis[1] / body_axis[0]))

            # -90 < atan(y/x) < 90, convert to 360 deg
            if body_axis[0] < 0:
                gt_heading_body_axis += 180
            elif body_axis[1] < 0:
                gt_heading_body_axis += 360

            # print('body_axis: ', body_axis, ', angle: ', gt_heading_body_axis)

            # estimated heading = calculated global aligned y-axis of the sensor
            imu_global_y_x = \
                imu_tracker.tracking_data[sensor_num].global_imu_coordinate_axes[
                    imu_index - imu_tracker.tracking_data[sensor_num].start_q_calc_index, 2][
                    0]
            imu_global_y_y = \
                imu_tracker.tracking_data[sensor_num].global_imu_coordinate_axes[
                    imu_index - imu_tracker.tracking_data[sensor_num].start_q_calc_index, 2][
                    1]
            estimated_heading_body_axis = math.degrees(math.atan(imu_global_y_y / imu_global_y_x))

            if imu_global_y_x < 0:
                estimated_heading_body_axis += 180
            elif imu_global_y_y < 0:
                estimated_heading_body_axis += 360

            diff = estimated_heading_body_axis - gt_heading_body_axis - offset

            # # calc gt heading, vertical to body axes, to campare with local imu z-axes
            # gt_heading_angle = gt_heading_body_axis - 90
            # if gt_heading_angle < 0:
            #     gt_heading_angle = 360 + gt_heading_angle
            #
            # # calc estimated heading
            # estimated_heading_angle = estimated_heading_body_axis - 90
            # if estimated_heading_angle < 0:
            #     estimated_heading_angle = 360 + estimated_heading_angle

            # subtract offset to calc the aligned heading
            estimated_heading_body_axis = estimated_heading_body_axis - offset

            if estimated_heading_body_axis > 360:
                estimated_heading_body_axis -= 360
            elif estimated_heading_body_axis < 0:
                estimated_heading_body_axis += 360

            estimated_heading[
                imu_index - imu_tracker.tracking_data[sensor_num].start_tracking_index] = (math.degrees(
                math.atan2(imu_global_y_y, imu_global_y_x)) + 360) % 360
            gt_heading[
                imu_index - imu_tracker.tracking_data[sensor_num].start_tracking_index] = (math.degrees(
                math.atan2(body_axis[1], body_axis[0])) + 360) % 360


            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360

            if imu_index == imu_tracker.tracking_data[sensor_num].start_tracking_index:
                # init offset
                # start assumption: no difference in angles
                offset = diff
                diff = 0
                print('offset ', offset)

            # q_align = Quaternion.quaternion_from_angle_and_axis(-offset, [0,0,1])
            # rotated_body_axis = q_align.rotate_v([imu_global_y_x, imu_global_y_y])

            # estimated_body_axis_new = math.degrees(math.atan(rotated_body_axis[1] / rotated_body_axis[0]))

            # if rotated_body_axis[0] < 0:
            #    estimated_body_axis_new += 180
            # elif rotated_body_axis[1] < 0:
            #    estimated_body_axis_new += 360

            # print('estimated: ', estimated_body_axis, ', new: ', estimated_body_axis_new, ', diff: ', diff)

            angle_diff[imu_index - imu_tracker.tracking_data[sensor_num].start_tracking_index] = diff

        mean = np.mean(angle_diff)
        mean_std = np.std(angle_diff)
        mean_abs = np.mean(abs(angle_diff))
        mean_abs_std = np.std(abs(angle_diff))
        max = np.max(angle_diff)
        min = np.min(angle_diff)
        max_abs = np.max(abs(angle_diff))
        logging.getLogger(__name__).info(
            'mean error: %f (std %f), mean_abs: %f (std %f), max: %f, min: %f, max_abs: %f', mean, mean_std, mean_abs,
            mean_abs_std,
            max, min, max_abs)

        if file is not None:
            config = imu_tracker.imu_data[sensor_num].description

            f = open(file, 'a+')
            f.write(
                '{:10} {:>5.2f} {:>5.2f} {:>5.2f} {:>5.2f} {:>5.2f} {:>5.2f} {:>5.2f}\n'.format(config, mean, mean_std,
                                                                                                mean_abs, mean_abs_std,
                                                                                                max, min,
                                                                                                max_abs))
            f.close()

        return [angle_diff, estimated_heading, gt_heading, offset]

    @staticmethod
    # calculate lifted (continouesly) heading angles for GT and IMU
    def calc_lifted_rotation_angles(imu_tracker, gt_left, gt_right, sensor_num=0):

        offset = 0

        gt_heading_x_y = []
        estimated_heading_x_y = []

        for imu_index in range(imu_tracker.tracking_data[sensor_num].start_tracking_index,
                               imu_tracker.tracking_data[sensor_num].end_tracking_index):
            camera_frame = imu_tracker.tracking_data[sensor_num].imu_to_camera[imu_index]

            body_axis = gt_left.get_position(camera_frame) - gt_right.get_position(camera_frame)

            gt_heading_body_axis = math.degrees(math.atan(body_axis[1] / body_axis[0]))
            gt_heading_x_y.append([body_axis[0], body_axis[1]])

            # print('body_axis: ', body_axis, ', angle: ', gt_heading_body_axis)

            # estimated heading = calculated global aligned y-axis of the sensor
            imu_global_y_x = \
                imu_tracker.tracking_data[sensor_num].global_imu_coordinate_axes[
                    imu_index - imu_tracker.tracking_data[sensor_num].start_q_calc_index, 2][
                    0]
            imu_global_y_y = \
                imu_tracker.tracking_data[sensor_num].global_imu_coordinate_axes[
                    imu_index - imu_tracker.tracking_data[sensor_num].start_q_calc_index, 2][
                    1]
            estimated_heading_body_axis = math.degrees(math.atan(imu_global_y_y / imu_global_y_x))
            estimated_heading_x_y.append([imu_global_y_x, imu_global_y_y])

            if imu_index == imu_tracker.tracking_data[sensor_num].start_tracking_index:
                # init offset

                # -90 < atan(y/x) < 90, convert to 360 deg
                if body_axis[0] < 0:
                    gt_heading_body_axis += 180
                elif body_axis[1] < 0:
                    gt_heading_body_axis += 360

                if imu_global_y_x < 0:
                    estimated_heading_body_axis += 180
                elif imu_global_y_y < 0:
                    estimated_heading_body_axis += 360

                diff = estimated_heading_body_axis - gt_heading_body_axis - offset

                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360

                offset = diff
                print('offset ', offset)

        # lift angles

        last = 180
        gt_heading = []
        for x, y in gt_heading_x_y:
            angle = (math.degrees(math.atan2(y, x)) + 360) % 360
            while angle < last - 180: angle += 360
            while angle > last + 180: angle -= 360
            last = angle
            gt_heading.append(angle)

        last = 180
        estimated_heading = []
        for x, y in estimated_heading_x_y:
            angle = (math.degrees(math.atan2(y, x)) + 360) % 360
            while angle < last - 180: angle += 360
            while angle > last + 180: angle -= 360
            last = angle
            estimated_heading.append(angle - offset)

        return [estimated_heading, gt_heading, offset]

    @staticmethod
    # compare calculated heading angle (aligned global z-axis) with angle velocity vector calculated with petrack
    def calculate_rotation_diff(imu_tracker, file=None, sensor_num=0):

        logging.getLogger(__name__).info('Validating rotation')

        angle_diff = np.zeros(imu_tracker.tracking_data[sensor_num].end_tracking_index - imu_tracker.tracking_data[
            sensor_num].start_tracking_index)
        gt_heading = np.zeros(imu_tracker.tracking_data[sensor_num].end_tracking_index - imu_tracker.tracking_data[
            sensor_num].start_tracking_index)

        for imu_index in range(imu_tracker.tracking_data[sensor_num].start_tracking_index,
                               imu_tracker.tracking_data[sensor_num].end_tracking_index):
            camera_frame = imu_tracker.tracking_data[sensor_num].imu_to_camera[imu_index]


            # velocity provides heading information
            v_gt = imu_tracker.ground_truth_trajectory.get_velocity(camera_frame)

            heading_angle = math.degrees(math.atan(v_gt[1] / v_gt[0]))

            # -90 < atan(y/x) < 90, convert to 360 deg
            if v_gt[0] < 0:
                heading_angle += 180
            elif v_gt[1] < 0:
                heading_angle += 360

            gt_heading[imu_index - imu_tracker.tracking_data[sensor_num].start_tracking_index] = heading_angle

            #print('body_axis: ', body_axis, ', angle: ', heading_angle)

            # estimated heading = calculated global aligned z-axis of the sensor
            imu_global_y_x = \
                imu_tracker.tracking_data[sensor_num].aligned_global_imu_coordinate_axes[
                    imu_index - imu_tracker.tracking_data[sensor_num].start_q_calc_index, 3][
                    0]
            imu_global_y_y = \
                imu_tracker.tracking_data[sensor_num].aligned_global_imu_coordinate_axes[
                    imu_index - imu_tracker.tracking_data[sensor_num].start_q_calc_index, 3][
                    1]
            estimated_body_axis = math.degrees(math.atan(imu_global_y_y / imu_global_y_x))

            if imu_global_y_x < 0:
                estimated_body_axis += 180
            elif imu_global_y_y < 0:
                estimated_body_axis += 360

            diff = estimated_body_axis - heading_angle

            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360

            angle_diff[imu_index - imu_tracker.tracking_data[sensor_num].start_tracking_index] = diff

        mean = np.mean(angle_diff)
        mean_abs = np.mean(abs(angle_diff))
        max = np.max(angle_diff)
        min = np.min(angle_diff)
        logging.getLogger(__name__).info('mean error: %f, mean_abs: %f, max: %f, min: %f', mean, mean_abs, max, min)

        if file is not None:
            config = imu_tracker.imu_data[sensor_num].description

            f = open(file, 'a+')
            f.write('#' + config + '\n')
            #format: mean(estimated-gt) mean(abs(estimated-gt)) max(diff) min(diff)
            f.write('{:>5.2f} {:>5.2f} {:>5.2f} {:>5.2f}\n'.format(mean, mean_abs, max, min))
            f.close()

        return angle_diff, gt_heading

    @staticmethod
    # compare calculated heading angle (aligned global z-axis) with the angle of the given vector
    # given vector can be: main movement direction, aligned with geometry, ...
    def validate_rotation_diff_to_vector(imu_tracker, directional_v, file=None, sensor_num=0):

        logging.getLogger(__name__).info('Validating rotation')

        angle_diff = np.zeros(imu_tracker.tracking_data[sensor_num].end_tracking_index - imu_tracker.tracking_data[
            sensor_num].start_tracking_index)
        gt_heading = np.zeros(imu_tracker.tracking_data[sensor_num].end_tracking_index - imu_tracker.tracking_data[
            sensor_num].start_tracking_index)

        for imu_index in range(imu_tracker.tracking_data[sensor_num].start_tracking_index,
                               imu_tracker.tracking_data[sensor_num].end_tracking_index):
            heading_angle = math.degrees(math.atan(directional_v[1] / directional_v[0]))

            # -90 < atan(y/x) < 90, convert to 360 deg
            if directional_v[0] < 0:
                heading_angle += 180
            elif directional_v[1] < 0:
                heading_angle += 360

            #print('COMPARE: ', heading_angle, ' should be ', math.degrees(math.atan2(directional_v[1], directional_v[0])))

            gt_heading[imu_index - imu_tracker.tracking_data[sensor_num].start_tracking_index] = heading_angle

            # print('body_axis: ', body_axis, ', angle: ', heading_angle)

            # estimated heading = calculated global aligned z-axis of the sensor
            imu_global_y_x = \
                imu_tracker.tracking_data[sensor_num].aligned_global_imu_coordinate_axes[
                    imu_index - imu_tracker.tracking_data[sensor_num].start_q_calc_index, 3][
                    0]
            imu_global_y_y = \
                imu_tracker.tracking_data[sensor_num].aligned_global_imu_coordinate_axes[
                    imu_index - imu_tracker.tracking_data[sensor_num].start_q_calc_index, 3][
                    1]
            estimated_body_axis = math.degrees(math.atan(imu_global_y_y / imu_global_y_x))

            if imu_global_y_x < 0:
                estimated_body_axis += 180
            elif imu_global_y_y < 0:
                estimated_body_axis += 360

            #print('COMPARE: ', estimated_body_axis, ' should be ', math.degrees(math.atan2(imu_global_y_y, imu_global_y_x)))

            diff = estimated_body_axis - heading_angle

            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360


            print('imu-gt=: ', estimated_body_axis, heading_angle, diff)

            angle_diff[imu_index - imu_tracker.tracking_data[sensor_num].start_tracking_index] = diff

        mean = np.mean(angle_diff)
        mean_abs = np.mean(abs(angle_diff))
        max = np.max(angle_diff)
        min = np.min(angle_diff)
        logging.getLogger(__name__).info('mean error: %f, mean_abs: %f, max: %f, min: %f', mean, mean_abs, max, min)

        if file is not None:
            config = imu_tracker.imu_data[sensor_num].description

            f = open(file, 'a+')
            f.write('#' + config + '\n')
            # format: mean(estimated-gt) mean(abs(estimated-gt)) max(diff) min(diff)
            f.write('{:>5.2f} {:>5.2f} {:>5.2f} {:>5.2f}\n'.format(mean, mean_abs, max, min))
            f.close()

        return angle_diff, gt_heading

    @staticmethod
    def validate_rotation_file(file_name):

        if not os.path.exists(file_name):
            logging.getLogger(__name__).error('%s does not exist', file_name)
            sys.exit()

        # open, process and close f
        with open(file_name):
            data = np.loadtxt(file_name)

        logging.getLogger(__name__).info(
            'mean_mean: %f, mean_mean_abs: %f (stdev: %f), mean_max: %f (stdev: %f), mean_min %f for %i data sets',
            np.mean(data[:, 0]), np.mean(data[:, 1]), np.std(data[:, 1]), np.mean(data[:, 2]), np.std(data[:, 2]),
            np.mean(data[:, 3]), len(data[:, 0]))

    @staticmethod
    def validate_acc_adaption(imu_tracker, file_name, sensor_num=0):

        corrected_samples = imu_tracker.tracking_data[sensor_num].acc_adaption_corrected_samples
        num_samples = imu_tracker.tracking_data[sensor_num].end_tracking_index - imu_tracker.tracking_data[
            sensor_num].start_tracking_index

        diffs = []
        for i in range(1, len(corrected_samples)):
            diffs.append(corrected_samples[i] - corrected_samples[i - 1])

        if len(diffs) > 0:
            mean = np.mean(diffs)
            max = np.max(diffs)
            min = np.min(diffs)

            if file_name is not None:
                config = imu_tracker.imu_data[sensor_num].description

                f = open(file_name, 'a+')
                f.write('#' + config + '\n')
                #format: num samples, num adaptions, mean diff between adaptions, max diff between adaptions, min diff between adaptions
                f.write(
                    '{:>5d} {:>5d} {:>5.2f} {:>5d} {:>5d}\n'.format(num_samples, len(corrected_samples), mean, max,
                                                                        min))
                f.close()

    @staticmethod
    def validate_acc_adaption_file(file_name):

        if not os.path.exists(file_name):
            logging.getLogger(__name__).error('%s does not exist', file_name)
            sys.exit()

        # open, process and close f
        with open(file_name):
            data = np.loadtxt(file_name)

        corrections_percentage = data[:,1] / data[:,0]

        logging.getLogger(__name__).info(
            'corrected_samples/all_samples: %f (stdev: %f), mean_mean: %f , max_max: %f, mean_max: %f, min_min: %f, mean_min: %f for %i datasets',
            np.mean(corrections_percentage), np.std(corrections_percentage), np.mean(data[:, 2]), np.max(data[:, 3]),
            np.mean(data[:, 3]), np.min([data[:, 4]]),
            np.mean(data[:, 4]), len(corrections_percentage))

    # TODO update
    @staticmethod
    def calc_average_diff(traj1, traj2):

        diff_sum = 0
        print(traj1.pos[:])
        shift = int(len(traj1.pos[:]) / len(traj2.pos[:]))
        # 8/2 = 4
        # 2/8 = 1/4
        for i in range(0, len(traj2.pos[:])):
            diff = traj2.pos[i][0:2] - traj1.pos[i * shift][0:2]
            diff_sum += LA.norm(diff)

        diff_sum /= len(traj2.pos[:])
        # TODO shift time start
        print('average_distance for ', len(traj2.pos[:]), ' positions: ', diff_sum)
        return diff_sum


    @staticmethod
    def calc_diff_in_position_2D(imu_tracker, file_name=None, sensor_num=0):

        camera_frame_start = imu_tracker.tracking_data[sensor_num].imu_to_camera[imu_tracker.tracking_data[sensor_num].start_tracking_index]
        camera_frame_end = imu_tracker.tracking_data[sensor_num].imu_to_camera[imu_tracker.tracking_data[sensor_num].end_tracking_index]

        diff = []

        print('calc diff for sensor ', sensor_num)

        for frame in range(camera_frame_start, camera_frame_end):

            imu_sample = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame]

            # imu sample could be out of range when tracking range was corrected since different num of sample for both sensors
            if imu_tracker.tracking_data[sensor_num].start_tracking_index <= imu_sample <= imu_tracker.tracking_data[
                sensor_num].end_tracking_index:
                diff.append(LA.norm((imu_tracker.ground_truth_trajectory.get_position(frame)[0:2] -
                                     imu_tracker.imu_data[sensor_num].trajectory.pos[
                                         imu_sample - imu_tracker.tracking_data[sensor_num].start_tracking_index][
                                     0:2])))
                #print('frame: ', frame, ', sample: ', imu_sample, ', imu_index: ', imu_sample-imu_tracker.tracking_data[sensor_num].start_tracking_index, ', diff: ', diff[-1], ', pos: ', imu_tracker.ground_truth_trajectory.get_position(frame)[0:2],imu_tracker.imu_data[sensor_num].trajectory.pos[imu_sample-imu_tracker.tracking_data[sensor_num].start_tracking_index][0:2] )

        logging.getLogger(__name__).info('num frames: %i, mean_diff: %f, max_diff: %f',len(diff), np.mean(diff), np.max(diff))

        if file_name is not None:
            config = imu_tracker.imu_data[sensor_num].description

            f = open(file_name, 'a+')
            f.write('#' + config + '\n')
            # format: num_frames, mean(diff), max(diff)
            f.write('{:>5d} {:>5.2f} {:>5.2f} \n'.format(len(diff), np.mean(diff), np.max(diff)))
            f.close()

        return diff

    @staticmethod
    def validate_diff_in_position_2D(file_name, frame_rate):

        if not os.path.exists(file_name):
            logging.getLogger(__name__).error('%s does not exist', file_name)
            sys.exit()

        # open, process and close f
        with open(file_name):
            data = np.loadtxt(file_name)

        logging.getLogger(__name__).info(
            'mean_time: %f s, mean_mean: %f (stdev: %f), mean_max: %f (stdev: %f), max_max: %f for %i datasets',
            np.mean(data[:, 0]) / frame_rate, np.mean(data[:, 1]), np.std(data[:, 1]), np.mean(data[:, 2]),
            np.std(data[:, 2]), np.max(data[:,
                                       2]), len(data[:, 0]))

    @staticmethod
    def dotproduct(v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))

    @staticmethod
    def length(v):
        return math.sqrt(Analysis.dotproduct(v, v))

    @staticmethod
    def angle(v1, v2):
        return math.acos(Analysis.dotproduct(v1, v2) / (Analysis.length(v1) * Analysis.length(v2)))

    @staticmethod
    def write_standstill_data_to_file(file, imu_tracker, sensor_name, frame):

        gt = imu_tracker.ground_truth_trajectory

        # check if file not exists yet --> write header
        if not os.path.exists(file):
            f_new = open(file, "w+")
            f_new.write(
                "# calculated acc in standstill and angle to gravity vector to validate rotation tracking quality" + '\n')
            f_new.write("# id, sensor name, frame, global acc x, global acc y, global acc z, angle diff\n")
            f_new.close()

        f_new = open(file, "a+")
        frame_to_sample = imu_tracker.tracking_data[0].camera_to_imu
        imu_sample = frame_to_sample[frame]
        acc = imu_tracker.tracking_data[0].acc_global[imu_sample]
        gravity = [0, 0, 9.81]

        # calc angle between acc and gravity
        error_angle = math.degrees(Analysis.angle(acc, gravity))

        print('acc: ', acc, ', angle: ', error_angle)

        f_new.write("%s \t %s \t %s \t %f \t %f \t %f \t %f\n" % (
        gt.identifier, sensor_name, frame, acc[0], acc[1], acc[2], error_angle))
        f_new.close()
