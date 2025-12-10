import os, sys, argparse, yaml, math
import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from math import ceil 
import wfdb
from imgaug import augmenters as iaa

BIT_NAN_16 = -(2.**15)

def read_config_file(config_file):
    """Read YAML config file

    Args:
        config_file (str): Complete path to the config file
    
    Returns:
        configs (dict): Returns dictionary with all the configs
    """
    with open(config_file) as f:
            yamlObject = yaml.safe_load(f)

    args = dict()
    for key in yamlObject:
        args[key] = yamlObject[key]

    return args

def find_records(folder, output_dir):
    header_files = list()
    recording_files = list()

    for root, directories, files in os.walk(folder):
        files = sorted(files)
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == '.mat':
                record = os.path.relpath(os.path.join(root, file.split('.')[0] + '.mat'), folder)
                hd = os.path.relpath(os.path.join(root, file.split('.')[0] + '.hea'), folder)
                recording_files.append(record)
                header_files.append(hd)
            if extension == '.dat':
                record = os.path.relpath(os.path.join(root, file.split('.')[0] + '.dat'), folder)
                hd = os.path.relpath(os.path.join(root, file.split('.')[0] + '.hea'), folder)
                header_files.append(hd)
                recording_files.append(record)
    
    if recording_files == []:
        raise Exception("The input directory does not have any WFDB compatible ECG files, please re-check the folder!")


    for file in recording_files:
        f, ext = os.path.splitext(file)
        f1 = f.split('/')[:-1]
        f1 = '/'.join(f1)

        if os.path.exists(os.path.join(output_dir, f1)) == False:
            os.makedirs(os.path.join(output_dir, f1))

    return header_files, recording_files


def find_files(data_directory):
    header_files = list()
    recording_files = list()

    for f in sorted(os.listdir(data_directory)):

        if(os.path.isdir(os.path.join(data_directory, f))):
            
            for file in sorted(os.listdir(os.path.join(data_directory,f))):
                root, extension = os.path.splitext(file)
                
                if not root.startswith('.'):

                    if extension=='.mat':
                        header_file = os.path.join(os.path.join(data_directory,f), root + '.hea')
                        recording_file = os.path.join(os.path.join(data_directory,f), root + '.mat')

                        if os.path.isfile(header_file) and os.path.isfile(recording_file):
                            header_files.append(header_file)
                            recording_files.append(recording_file)

                    if extension=='.dat':
                        header_file = os.path.join(os.path.join(data_directory,f), root + '.hea')
                        recording_file = os.path.join(os.path.join(data_directory,f), root + '.dat')

                        if os.path.isfile(header_file) and os.path.isfile(recording_file):
                            header_files.append(header_file)
                            recording_files.append(recording_file)
                            
        else:

            root, extension = os.path.splitext(f)

            if not root.startswith('.'):
                #Based on the recording format, we save the file names differently
                if extension=='.mat':
                    header_file = os.path.join(data_directory, root + '.hea')
                    recording_file = os.path.join(data_directory, root + '.mat')
                    if os.path.isfile(header_file) and os.path.isfile(recording_file):
                        header_files.append(header_file)
                        recording_files.append(recording_file)

                if extension=='.dat':
                    header_file = os.path.join(data_directory, root + '.hea')
                    recording_file = os.path.join(data_directory, root + '.dat')
                    if os.path.isfile(header_file) and os.path.isfile(recording_file):
                        header_files.append(header_file)
                        recording_files.append(recording_file)

    return header_files, recording_files

def load_header(header_file):
    with open(header_file, 'r') as f:
        header = f.read()
    return header

# Load recording file as an array.
def load_recording(recording_file, header=None,key='val'):
    rootname,extension = os.path.splitext(recording_file)
    #Load files differently based on file format
    if extension=='.dat':
        recording = wfdb.rdrecord(rootname)
        return recording.p_signal
    if extension=='.mat':
        recording = loadmat(recording_file)[key]
    return recording

# Get leads from header.
def get_leads(header):
    leads = list()
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            num_leads = int(entries[1])
        elif i<=num_leads:
            leads.append(entries[-1])
        else:
            break
    return tuple(leads)

# Get frequency from header.
def get_frequency(header):
    frequency = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                frequency = l.split(' ')[2]
                if '/' in frequency:
                    frequency = float(frequency.split('/')[0])
                else:
                    frequency = float(frequency)
            except:
                pass
        else:
            break
    return frequency

# Get analog-to-digital converter (ADC) gains from header.
def get_adc_gains(header, leads):
    adc_gains = np.zeros(len(leads))
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            num_leads = int(entries[1])
        elif i<=num_leads:
            current_lead = entries[-1]
            if current_lead in leads:
                j = leads.index(current_lead)
                try:
                    adc_gains[j] = float(entries[2].split('/')[0])
                except:
                    pass
        else:
            break
    return adc_gains


def truncate_signal(signal,sampling_rate,length_in_secs):
    signal=signal[0:int(sampling_rate*length_in_secs)]
    return signal

def create_signal_dictionary(signal,full_leads):
    record_dict = {}
    for k in range(len(full_leads)):
        record_dict[full_leads[k]] = signal[k]
        
    return record_dict

def standardize_leads(full_leads):
    full_leads_array = np.asarray(full_leads)
    
    for i in np.arange(len(full_leads_array)):
        if(full_leads_array[i].upper() not in ('AVR','AVL','AVF')):
            full_leads_array[i] = full_leads_array[i].upper()
        else:
            if(full_leads_array[i].upper()=='AVR'):
                full_leads_array[i] = 'aVR'
            elif(full_leads_array[i].upper()=='AVL'):
                full_leads_array[i] = 'aVL'
            else:
                 full_leads_array[i] = 'aVF'
    return full_leads_array

def rotate_bounding_box(box, origin, angle):
    angle = math.radians(angle)

    transformation = np.ones((2, 2))
    transformation[0][0] = math.cos(angle)
    transformation[0][1] = math.sin(angle)
    transformation[1][0] = -math.sin(angle)
    transformation[1][1] = math.cos(angle)

    new_origin = np.ones((1, 2))
    new_origin[0, 0] = -origin[0]*math.cos(angle) + origin[1]*math.sin(angle)
    new_origin[0, 1] = -origin[0]*math.sin(angle) - origin[1]*math.cos(angle)
    origin = np.reshape(origin, (1, 2))

    transformed_box = np.matmul(box, transformation)    
    transformed_box += origin + new_origin 

    return transformed_box

def read_leads(leads):

    lead_bbs = []
    text_bbs = []
    startTimeStamps = []
    endTimeStamps = []
    labels = []
    plotted_pixels = []
    for i, line in enumerate(leads):
        labels.append(leads[i]['lead_name'])
        st_time_stamp = leads[i]['start_sample']
        startTimeStamps.append(st_time_stamp)
        end_time_stamp = leads[i]['end_sample']
        endTimeStamps.append(end_time_stamp)
        plotted_pixels.append(leads[i]['plotted_pixels'])

        key = "lead_bounding_box"
        if key in leads[i].keys():
            parts = leads[i][key]
            point1 = [parts['0'][0], parts['0'][1]]
            point2 = [parts['1'][0], parts['1'][1]]
            point3 = [parts['2'][0], parts['2'][1]]
            point4 = [parts['3'][0], parts['3'][1]]
            box = [point1, point2, point3, point4]
            lead_bbs.append(box)

        key = "text_bounding_box"
        if key in leads[i].keys():
            parts = leads[i][key]
            point1 = [parts['0'][0], parts['0'][1]]
            point2 = [parts['1'][0], parts['1'][1]]
            point3 = [parts['2'][0], parts['2'][1]]
            point4 = [parts['3'][0], parts['3'][1]]
            box = [point1, point2, point3, point4]
            text_bbs.append(box)

    if len(lead_bbs) != 0:
        lead_bbs = np.array(lead_bbs)
    if len(text_bbs) != 0:
        text_bbs = np.array(text_bbs)

    return lead_bbs, text_bbs, labels, startTimeStamps, endTimeStamps, plotted_pixels

def convert_bounding_boxes_to_dict(lead_bboxes, text_bboxes, labels, startTimeList = None, endTimeList = None, plotted_pixels_dict=None):
    leads_ds = []

    for i in range(len(labels)):
        current_lead_ds = dict()
        if len(lead_bboxes) != 0:
            new_box = dict()
            box = lead_bboxes[i]
            new_box[0] = [round(box[0][0]), round(box[0][1])]
            new_box[1] = [round(box[1][0]), round(box[1][1])]
            new_box[2] = [round(box[2][0]), round(box[2][1])]
            new_box[3] = [round(box[3][0]), round(box[3][1])]
            current_lead_ds["lead_bounding_box"] = new_box

        if len(text_bboxes) != 0:
            new_box = dict()
            box = text_bboxes[i]
            new_box[0] = [round(box[0][0]), round(box[0][1])]
            new_box[1] = [round(box[1][0]), round(box[1][1])]
            new_box[2] = [round(box[2][0]), round(box[2][1])]
            new_box[3] = [round(box[3][0]), round(box[3][1])]
            current_lead_ds["text_bounding_box"] = new_box

        current_lead_ds["lead_name"] = labels[i]
        current_lead_ds["start_sample"] = startTimeList[i]
        current_lead_ds["end_sample"] = endTimeList[i]
        current_lead_ds["plotted_pixels"] = [[plotted_pixels_dict[i][j][0], plotted_pixels_dict[i][j][1]] for j in range(len(plotted_pixels_dict[i]))]
        leads_ds.append(current_lead_ds)

    return leads_ds


def convert_mm_to_volts(mm):
    return float(mm/10)

def convert_mm_to_seconds(mm):
    return float(mm*0.04)

def convert_inches_to_volts(inches):
    return float(inches*2.54)

def convert_inches_to_seconds(inches):
    return float(inches*1.016)

def write_wfdb_file(ecg_frame, filename, rate, header_file, write_dir, full_mode, mask_unplotted_samples):
    full_header = load_header(header_file)
    full_leads = get_leads(full_header)
    full_leads = standardize_leads(full_leads)

    lead_step = 10.0
    samples = len(ecg_frame[full_mode])
    array = np.zeros((1, samples))

    leads = []
    header_name, extn = os.path.splitext(header_file)
    header = wfdb.rdheader(header_name)

    for i, lead in enumerate(full_leads):
        leads.append(lead)
        if lead == full_mode:
            lead = 'full' + lead
        adc_gn = header.adc_gain[i]

        arr = ecg_frame[lead]
        arr = np.array(arr)
        arr[np.isnan(arr)] = BIT_NAN_16/adc_gn
        arr = arr.reshape((1, arr.shape[0]))
        array = np.concatenate((array, arr),axis = 0)
    
    head, tail  = os.path.split(filename)
    
    array = array[1:]
    wfdb.wrsamp(record_name = tail, 
                fs = rate, units = header.units,
                sig_name = leads, p_signal = array.T, fmt = header.fmt,
                adc_gain = header.adc_gain, baseline = header.baseline, 
                base_time = header.base_time, base_date = header.base_date, 
                write_dir = write_dir, comments = header.comments)

def get_lead_pixel_coordinate(leads):

    pixel_coordinates = dict()

    for i in range(len(leads)):
        leadName = leads[i]["lead_name"]
        plotted_pixels = np.array(leads[i]["plotted_pixels"])
        pixel_coordinates[leadName] = plotted_pixels

    return pixel_coordinates


def rotate_points(pixel_coordinates, origin, angle):
    rotates_pixel_coords = []
    angle = math.radians(angle)
    transformation = np.ones((2, 2))
    transformation[0][0] = math.cos(angle)
    transformation[0][1] = math.sin(angle)
    transformation[1][0] = -math.sin(angle)
    transformation[1][1] = math.cos(angle)

    new_origin = np.ones((1, 2))
    
    new_origin[0, 0] = -origin[0]*math.cos(angle) + origin[1]*math.sin(angle)
    new_origin[0, 1] = -origin[0]*math.sin(angle) - origin[1]*math.cos(angle)
    origin = np.reshape(origin, (1, 2))
    

    for i in range(len(pixel_coordinates)):
        pixels_array = pixel_coordinates[i]
        transformed_matrix = np.matmul(pixels_array, transformation)
        transformed_matrix += origin + new_origin 
        rotates_pixel_coords.append(np.round(transformed_matrix, 2))
        
    return rotates_pixel_coords
