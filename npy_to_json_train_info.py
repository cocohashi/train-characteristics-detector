"""
Authors: Felipe, Hasier

Refactored to use python 3.9

Date 15/04/2024
"""

import os
import json
import numpy as np
import time
from scipy import signal
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import traceback
from pathlib import Path

# -------------------------------------------------------------------------------------------------------------------
import logging

logger = logging.getLogger(__name__)
logger.propagate = False
handler = logging.StreamHandler()
# handler = logging.FileHandler('main.log')
logger.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
# -----------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Import get_train_characteristics
file_path = Path(__file__).resolve()
parent_path, root_path = file_path.parent, file_path.parents[1]

sys.path.append(str(parent_path))

from main import get_train_characteristics


# -----------------------------------------------------------------------------

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return np.around(obj, 4).tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def get_valid_npy_filenames(data_folder):
    '''
    Function to get the valid filenames from a single.

    Parameters
    ---------
    data_folder    : str
        path to the data folder.

    Returns
    -------
    Returns a dictionary where each key-value is an output.

    output_dict: dict
        valid_filename_list: list
            list containing the valid filenames within the folder. 
        separator_str : list
            string containing the filename separator. Can be '.npy' or '_0.npy'
    '''

    filename_list = os.listdir(data_folder)
    separator_str = '.npy'

    valid_filename_list = []

    # select only valid .npy files, store valid filenames in valid_filename_list: 
    for filename in filename_list:

        not_indexes_file = "indexes" not in filename
        not_score_file = "score" not in filename
        not_state_file = "state" not in filename
        numpy_file = ".npy" in filename

        valid_file_condition = all([not_indexes_file, not_score_file, not_state_file, numpy_file])

        if valid_file_condition == True:
            valid_filename_list.append(filename)

    # get the correct string separator. some .npy files are saved with and extra _0.npy at the end of the filename:

    try:
        separated_str = valid_filename_list[0].split(separator_str)[0].split('_')

        # H_M_S_0.npy:  
        if len(separated_str) == 4:
            separator_str = '_0.npy'

        output_dict = {}
        output_dict['valid_filename_list'] = valid_filename_list
        output_dict['separator_str'] = separator_str

        return output_dict

    except:

        logger.info('Error: no valid filenames found in folder')

        return


def create_save_dict(data, d_sample, t_sample, start_key=0, train_info={}, str_train_datetime=""):
    temporal_samples = data.shape[1]

    # time and distance vector creation: 
    spatial_samples = data.shape[0]
    temporal_samples = data.shape[1]

    d_vector = np.arange(0, spatial_samples) * d_sample * 10
    t_vector = np.arange(0, temporal_samples) * t_sample

    # dictionary creation: 
    save_dict = {}

    save_dict["position"] = d_vector
    save_dict["measurements"] = {}

    # TODO: Do not hardcode this 
    new_temporal_samples = 30002
    data_npy = np.zeros(shape=(new_temporal_samples, spatial_samples))

    for i in range(0, new_temporal_samples):
        save_dict["measurements"][str(i + start_key)] = {}
        save_dict["measurements"][str(i + start_key)]["timestamp"] = t_vector[i]
        save_dict["measurements"][str(i + start_key)]["strain"] = data[:, i]
        data_npy[i, :] = data[:, i]

    #    for i in range(0,new_temporal_samples):
    #        data_npy[i,:] = save_dict["measurements"][str(i)]["strain"]

    #    data_shape = data.shape
    logger.info(data_npy.shape)
    # get_train_characteristics: 
    train_dict = get_train_characteristics(data_npy)
    train_info = train_dict['train-char']
    train_info['datetime'] = str_train_datetime

    # datetime

    logger.info(f"train-info: {train_info}")

    #    logger.info(train_info.keys())
    train_info_str = {k: str(v) for k, v in train_info.items()}
    save_dict['info'] = train_info_str

    return save_dict


def create_folder(route_folder, folder_name):
    folder_name_and_route = "%s/%s" % (route_folder, folder_name)

    if not os.path.exists(folder_name_and_route):
        os.mkdir(folder_name_and_route)

    return folder_name_and_route


def load_and_concatenate_data(day_folder, str_timestamp, phase_to_strain):
    # load npy data:
    filename_aux = '%s/%s' % (day_folder, str_timestamp)

    data_0 = np.load('%s_0.npy' % filename_aux)
    data_1 = np.load('%s_1.npy' % filename_aux)
    data_2 = np.load('%s_2.npy' % filename_aux)

    # concatenate over temporal axis: 
    data = np.concatenate((data_0, data_1, data_2), axis=1)

    # phase to strain conversion: 
    data = np.array(data, dtype=float) * phase_to_strain * 1e6

    return data


# def get_train_characteristics(data):
#    
#    return {}


def high_pass_filter(s, dt, fc):
    # --------------------------------------------------------------------------
    # Description: Code to apply a zero-phase high-pass filtering to signal.  
    # Note:        Each signal is assumed to be in a column of the matrix s. 
    # Inputs: 
    #     s  : Signal matrix.
    #     dt : Sampling period of signals. 
    #     fc : Filter cutoff frequency. 
    # Outputs: 
    #     filtered: Filtered signal matrix.  
    # --------------------------------------------------------------------------

    fs = 1 / dt
    b, a = signal.butter(4, fc, 'hp', fs=fs)

    filtered = signal.filtfilt(b, a, s, axis=0)
    filtered = filtered - np.mean(filtered, axis=0)

    return filtered


def get_report_json(train_info, filename, report_json={}):
    logger.info('get report:: train_info:: %s' % train_info)
    # get event    
    event = train_info['event']
    report_event_list = [event for event in report_json['event'].keys()]
    logger.info('report_event_list: %s' % report_event_list)

    # get train-id
    train_id = train_info['train-id']
    train_ids_list = [train_id for train_id in report_json['train-id'].keys()]
    logger.info('train_ids_list: %s' % train_ids_list)

    for report_event_type in report_event_list:
        if event == report_event_type:
            actual_report_event = report_json.get('event')
            report_event_train = actual_report_event.get(report_event_type)
            report_event_train.append(filename)
            report_event_train.sort()
            actual_report_event.update({report_event_type: report_event_train})
            report_json.update({"event": actual_report_event})

    for report_train_id_type in train_ids_list:
        if train_id == report_train_id_type:
            actual_report_train_id = report_json.get('train-id')
            report_train = actual_report_train_id.get(report_train_id_type)
            report_train.append(filename)
            report_train.sort()
            actual_report_train_id.update({report_train_id_type: report_train})
            report_json.update({"train-id": actual_report_train_id})

    return report_json


def generate_json_from_numpy(npy_day_folder, json_day_folder, json_image_day_folder, str_timestamp, now):
    # acquisition input data (hardcoded by the moment):

    #    sample_rate = 1e9
    T_pulse_eff = 499.1494476513955  # [us]
    GL = 5  # [m]

    # phase to strain conversion parameters (hardcoded by the moment): 

    lambdax = 1550.12 * 1e-9  # [m]
    neff = 1.46  # refractive index
    #    d_sample = 3e8/(2*neff*sample_rate)   # sampling distance [m]
    xi = 0.78  # optoelectric coefficient

    phase_to_strain = lambdax / (4 * np.pi * neff * GL * xi)

    # load npy data: concatenate files with same str_timestamp and 
    # terminations _0.npy, _1.npy, _2.npy:
    data = load_and_concatenate_data(npy_day_folder, str_timestamp, phase_to_strain)

    # Get event datetime   
    str_values = str_timestamp.split("_")
    values = [int(value) for value in str_values]
    hour, minute, second = values[0], values[1], values[2]
    train_datetime = now.replace(hour=hour, minute=minute, second=second)
    str_train_datetime = train_datetime.strftime("%Y-%m-%d %H:%M:%S")
    logger.info('Event datetime: %s' % str_train_datetime)

    # numpy to json conversion: 
    json_dict = create_save_dict(data,
                                 d_sample=GL,
                                 t_sample=T_pulse_eff * 1e-6 * 2,
                                 train_info={},
                                 str_train_datetime=str_train_datetime)

    # save json: 
    json_filename = '%s/%s.json' % (json_day_folder, str_timestamp)
    logger.info('Saving JSON at path: %s' % json_filename)

    try:
        with open(json_filename, 'w') as f:
            json.dump(json_dict, f, cls=NumpyArrayEncoder)
    except Exception as e:
        logger.error('Error while saving json DATA  file: %s' % traceback.format_exc())

    logger.info('json file successfully created')

    # TODO: Helper characters: [],  {}

    # filter data to  plot: 

    save_filename = '%s/%s' % (json_image_day_folder, str_timestamp)

    filtered_phase = high_pass_filter(data.T, dt=0.001, fc=0.8)

    plt.figure()
    plt.imshow(filtered_phase, aspect='auto', origin='lower', interpolation='none',
               cmap='seismic', vmin=-0.05, vmax=0.05)

    plt.savefig(save_filename)
    plt.close()

    # save report info in report_files.json    
    report_filename = "report_files.json"
    report_path = os.path.join(json_day_folder, report_filename)
    logger.info("repor_path: %s" % report_path)
    logger.info(os.path.isfile(report_path))

    # report json
    train_info = json_dict['info']

    if not os.path.isfile(report_path):
        logger.info("Creating new report file...")
        report_json = {
            "event":
                {
                    "train": [],
                    "no-train": [],
                    "double-train": [],
                    "slow-train": [],
                    "unknown": []
                },
            "train-id":
                {
                    "1": [],
                    "2": [],
                    "3": [],
                    "4": [],
                    "5": []
                }
        }
        report_json = get_report_json(train_info=train_info,
                                      filename=str_timestamp,
                                      report_json=report_json)

        try:
            with open(report_path, 'w', encoding='utf-8') as file:
                json.dump(report_json, file)
        except Exception as e:
            logger.error('Error while saving json REPORT file: %s' % traceback.format_exc())

    else:
        logger.info("Updating report file")
        # get report json
        try:
            with open(report_path, 'r', encoding='utf-8') as file:
                report_json = json.loads(file.read())
            logger.info('REPORT file successfully loaded: %s' % report_json)
        except Exception as e:
            logger.error('Error saving json REPORT file: %s' % traceback.format_exc())

        # update report json        
        report_json = get_report_json(train_info=train_info,
                                      filename=str_timestamp,
                                      report_json=report_json)
        # save updated report json
        try:
            with open(report_path, 'w', encoding='utf-8') as file:
                json.dump(report_json, file)
        except Exception as e:
            logger.error('Error while saving json REPORT file: %s' % traceback.format_exc())

    return data


def main(npy_data_folder, json_main_save_folder, json_main_image_folder, now):
    (year, month, day) = (now.year, now.month, now.day)

    # set origin (npy) working folder:  
    npy_day_folder = '%s/%s/%s/%02d/%02d' % (npy_data_folder, 'phase', year, month, day)

    if not os.path.exists(npy_day_folder):

        logger.info("Folder :'%s' does not exist, try again later" % npy_day_folder)
        return

    else:

        filename_list = os.listdir(npy_day_folder)

        # set destination (json) working folder: 
        json_year_folder = create_folder(json_main_save_folder, year)
        json_month_folder = create_folder(json_year_folder, '%02d' % month)
        json_day_folder = create_folder(json_month_folder, '%02d' % day)

        # set destination (json) image folder: 
        json_image_year_folder = create_folder(json_main_image_folder, year)
        json_image_month_folder = create_folder(json_image_year_folder, '%02d' % month)
        json_image_day_folder = create_folder(json_image_month_folder, '%02d' % day)

        # check processed_files.json, create it if needed: 
        processed_files = 'processed_files.json'
        processed_files_full_path = '%s/%s' % (npy_day_folder, processed_files)

        if processed_files not in filename_list:
            with open(processed_files_full_path, 'w') as f:
                save_dict = dict(processed_files=[])
                json.dump(save_dict, f)
                logger.info('processed_files.json created')

        # load processed_files:
        processed_files_list = json.load(open(processed_files_full_path))['processed_files']
        processed_files_list_copy = processed_files_list.copy()
        new_files_list = []

        # get valid npy filenames: 
        output_dict = get_valid_npy_filenames(npy_day_folder)
        valid_filename_list = output_dict['valid_filename_list']
        separator_str = output_dict['separator_str']

        # separate only filenames ended with _0.npy: 
        str_timestamp_list = [filename.split(separator_str)[0] for filename in valid_filename_list if
                              separator_str in filename]

        # iterate over filenames and create json files if needed: 
        for str_timestamp in str_timestamp_list:

            # create json if needed: 
            if str_timestamp not in processed_files_list:
                logger.info('file: %s not found in processed files, generating json file:' % str_timestamp)

                last_buffer_file = '%s/%s_2.npy' % (npy_day_folder, str_timestamp)

                if os.path.exists(last_buffer_file):

                    generate_json_from_numpy(npy_day_folder, json_day_folder, json_image_day_folder, str_timestamp, now)

                    processed_files_list_copy.append(str_timestamp)
                    new_files_list.append(str_timestamp)

                else:

                    logger.info(
                        'Error: buffer not completed yet, mising file: %s_2.npy, try again later' % (str_timestamp))

        # update processed_files.json if there are changes: 
        if processed_files_list != processed_files_list_copy:

            logger.info('new files added: %s' % new_files_list)

            with open(processed_files_full_path, 'w') as f:
                save_dict = dict(processed_files=processed_files_list_copy)
                json.dump(save_dict, f)
                logger.info('processed_files.json updated')
        else:

            logger.info('No new files found')


if __name__ == '__main__':

    # input data: 
    npy_data_folder = '/mnt/Data/Preprocessed_phase/5_m_2_km_1_point_per_GL'
    json_main_save_folder = '/mnt/Data/Preprocessed_phase/json_files'
    json_main_image_folder = '/mnt/Data/Preprocessed_phase/alert_images'

    running = True

    while (running):

        now = datetime.now()
        #        (year, month, day) = (now.year,now.month,now.day)
        #        (year, month, day) = (2024,4,8)

        try:
            # execute function: 
            logger.info("-----------------------------------------------")
            logger.info('datetime: %s' % now.strftime('%Y-%m-%d %H:%M:%S'))
            logger.info('Numpy input path: %s' % npy_data_folder)
            logger.info('JSON output path: %s' % json_main_save_folder)
            logger.info('Waterfall images path: %s' % json_main_image_folder)
            main(npy_data_folder, json_main_save_folder, json_main_image_folder, now)
            logger.info("-----------------------------------------------")

        except Exception as e:
            logger.error(f'Error: {traceback.format_exc()}')
            logger.error('Error detected, re-launching code')

        time.sleep(10)
