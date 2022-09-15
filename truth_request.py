import json
from operator import truth
import os
import requests
import numpy as np
import datetime
from datetime import timedelta
import orekit
from orekit.pyhelpers import  setup_orekit_curdir
import auth
import truth_analysis as ta
from dateutil.parser import parse as date_parse
import statistics
from itertools import zip_longest
from utilities.aws_helper import AwsHelper
from utilities.api import Api
from utilities.od_logging import OptionalLog
import re
import time
import shutil
from matplotlib import pyplot as plt
import asyncio


def id_data(leo_id):
    """ID Data for a LeoLabs object to initialize the truth analysis object."""
    
    object_url = "".join([auth.api_url, '/catalog/objects/',leo_id])
    object_response = requests.get(object_url,headers=auth.headers)
    leolabs_id = object_response.json()["catalogNumber"] 
    norad_id = object_response.json()["noradCatalogNumber"]
    name = object_response.json()["name"]
    
    id_dict = {}
    id_dict['leolabs_id'] = leolabs_id
    id_dict['norad_id'] = norad_id
    id_dict['object_name'] = name
    return id_dict

def request_response(object_url):
    request_response = requests.get(object_url,headers=auth.headers)
    return request_response

async def async_id_data(leo_id):
    object_url = "".join([auth.api_url, '/catalog/objects/',leo_id])
    object_response = await asyncio.to_thread(request_response,object_url)
    leolabs_id = object_response.json()["catalogNumber"] 
    norad_id = object_response.json()["noradCatalogNumber"]
    name = object_response.json()["name"]
    
    id_dict = {}
    id_dict['leolabs_id'] = leolabs_id
    id_dict['norad_id'] = norad_id
    id_dict['object_name'] = name
    return id_dict

def propagation_dates_from_epoch(epoch):
    """Function handed an epoch (as a list) and returning start and end days, 1 day away from epoch"""
    
    epoch_in_dt = date_parse(epoch, ignoretz=True) # convert epoch to datetime
    start_time = epoch_in_dt - timedelta(seconds=86399) # start of propagations is 1 day before epoch
    end_time = epoch_in_dt + timedelta(seconds=86401) # end of propagations is 1 day after epoch
    # convert datetimes to strings
    start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    return start_time_str, end_time_str

def states_available_x_days_from_epoch(object_id,epoch,num_days):
    """Takes an object id and a date and returns all the states available for this target in a number of days prior to epoch"""
    
    end_time_str = datetime.datetime(epoch[0], epoch[1], epoch[2]).strftime("%Y-%m-%d") # end time is the epoch
    start_time = datetime.datetime(epoch[0], epoch[1], epoch[2]) - timedelta(days=num_days) # start time is num_days prior to epoch 
    start_time_str = start_time.strftime("%Y-%m-%d") 
    
    states_url = "".join([auth.api_url, '/catalog/objects/',object_id,'/states?startTime=',start_time_str,'&endTime=',end_time_str])
    
    states_response = requests.get(states_url,headers=auth.headers)
    
   
    states_list = states_response.json()["states"]
    
    if len(states_list)>0: # handling the exception where there are no states 

        states_arr = np.empty(3,dtype='<U32')

        for i in range(len(states_list)):
            states_arr = np.vstack((states_arr,[object_id,states_list[i]['id'],states_list[i]['timestamp']]))

        return states_arr[1:].astype('<U32')
    else:
        return None

def state_data(leo_id, start_date, end_date):
    """Pull all the states of a LeoLabs object between certain dates"""
    
    start_time = datetime.datetime(start_date[0], start_date[1], start_date[2]).strftime("%Y-%m-%d")
    end_time = datetime.datetime(end_date[0], end_date[1], end_date[2]).strftime("%Y-%m-%d")
    
    states_url = "".join([auth.api_url, '/catalog/objects/',leo_id,'/states?startTime=',start_time,'&endTime=',end_time])
    
    states_response = requests.get(states_url,headers=auth.headers)
    
    first_state_id = states_response.json()["states"][0]["id"]
    
    return states_response 

def RIC_covariance_of_propagations(leo_id, state_id, start_time, end_time, timestep):
    """Pull the RIC covariances out of the propagations """ 
    
    prop_with_ric_url = "".join([auth.api_url, '/catalog/objects/',leo_id,'/states/',state_id,'/propagations?startTime=',start_time,'&endTime=',end_time,'&timestep=',str(timestep),'&frame=RIC'])
    
    prop_with_ric_response = requests.get(prop_with_ric_url, headers=auth.headers)
    
    return prop_with_ric_response.json()["propagation"]

def propagation_of_state(leo_id, state_id, start_time, end_time, timestep):
    """Request state propagations between a start date and an end date and with certain timestep."""
    
    propagation_url = "".join([auth.api_url, '/catalog/objects/',leo_id,'/states/',state_id,'/propagations?startTime=',start_time,'&endTime=',end_time,'&timestep=',str(timestep)])
    
    propagation_response = requests.get(propagation_url, headers=auth.headers)
    
    return propagation_response.json()["propagation"]

async def async_propagation_of_state(leo_id, state_id, start_time, end_time, timestep):
    """Asynchronous request of state propagations between a start date and an end date and with certain timestep."""
    
    propagation_url = "".join([auth.api_url, '/catalog/objects/',leo_id,'/states/',state_id,'/propagations?startTime=',start_time,'&endTime=',end_time,'&timestep=',str(timestep)])
    
    propagation_response = await asyncio.to_thread(request_response,propagation_url)
    
    return propagation_response.json()["propagation"]

class PropagationsContainer():
    """Convenience container for propagations"""
    
    def __init__(self,timestamp,position,velocity,covariance):
        self.timestamp = timestamp
        self.position = position
        self.velocity = velocity
        self.covariance = covariance
        
class RicCovariancesContainer():
    """Convenience container for RIC covariances"""
    
    def __init__(self,covariance):
        self.covariance = covariance
               
def Ric_propagation_dict_to_container(ric_prop_dict):
    """Takes a single RIC propagation json dictionary and returns a RicCovariancesContainer object."""
    
    covariance = ric_prop_dict['covariance']
    
    return RicCovariancesContainer(covariance)

def propagation_dict_to_container(prop_dict):
    """Takes a single propagation list and returns a PropagationsContainer object."""
    
    timestamp = date_parse(prop_dict['timestamp'],ignoretz=True)
    position = prop_dict['position']
    velocity = prop_dict['velocity']
    covariance = prop_dict['covariance']
    
    return PropagationsContainer(timestamp,position,velocity,covariance)

def propagations_list(propagations_dict):
    """Takes a propagations dict and returns a list with PropagationContainer objects."""
    
    prop_list = []
    for i in range(len(propagations_dict)):
        prop_list.append(propagation_dict_to_container(propagations_dict[i]))
    
    return prop_list

def RIC_Covariances_list(ric_dict):
    """Takes a RIC propagations dict and returns a list of RicCovariancesContainer object."""
    
    ric_list = []
    for i in range(len(ric_dict)):
        ric_list.append(Ric_propagation_dict_to_container(ric_dict[i]).covariance)
        
    return ric_list

def extract_epochOffset(norm_errors_dict):
    """Takes a dictionary with normalized errors and extracts the epoch Offset and returns them into a list"""
    
    epoch_Offset_list = []
    
    for i in range(len(norm_errors_dict)):
        epoch_Offset_list.append(norm_errors_dict[i]['epochOffset'])
    
    return epoch_Offset_list

def extract_norm_error(norm_errors_dict, coord_ind):
    """Takes a dictionary with normalized errors and extracts any of the errors, based on the coord_ind"""
    
    error_list = []
    
    for i in range(len(norm_errors_dict)):
        error_list.append(norm_errors_dict[i]['vals'][coord_ind])
        
    return error_list

def z_score(d,MAD):
    return 0.6745*d/MAD

def cull_outliers(arr):
    """Function for culling outliers based on their mean squared distance from the median."""
    threshold = 3.0

    arr = np.array(arr)

    # Strip nans
    arr = arr[~np.isnan(arr)]

    diffs = np.sqrt((arr - np.median(arr))**2)

    MAD = np.median(diffs)

    return [y for x,y in zip(diffs,arr) if z_score(x,MAD) < threshold]

def extract_std_from_error_distributions(err_collection):
    """Takes a collection of error distributions and returns the standard deviation at each time step."""
    stdevs = []
    for i in range(len(err_collection)):
        stdevs.append(statistics.pstdev(cull_outliers(err_collection[i])))
    return stdevs

def aws_init(logger=None):
    """Initialize aws"""
    # Initialize API connection object
    api_client = Api.get_client(logger=logger)

    # Initialize AWS helper functions
    aws_helper = AwsHelper(logger=logger)
    
    return api_client, aws_helper 

def set_up_truth_directory_for_target(leolabs_id):
    """Prepares a local directory for storing ILRS truth files for particular ILRS target."""
    #try:
    #    shutil.rmtree('truth/' + str(leolabs_id)) # Remove target's directory if it already exists
    #except FileNotFoundError:
    #    pass

    base_truth_directory = 'truth/'
    try:
        os.mkdir(base_truth_directory)
    except OSError:
        pass

    truth_directory = base_truth_directory + str(leolabs_id)
    try:
        os.mkdir(truth_directory)
    except OSError:
        pass

    return truth_directory

def get_truth_file_list(epoch_date, norad_id, num_days):
    """Builds list of S3 truth files for the current object that match the date range of interest."""
    
    epoch_date_in_dt = datetime.datetime(epoch_date[0],epoch_date[1],epoch_date[2]) # we need this date in datetime
    
    def string_from_date(dt):
        return str(dt.year)[-2:] + '%02d' % (dt.month,) + '%02d' % (dt.day,)

    dates_of_interest = [(epoch_date_in_dt - timedelta(days=i)) for i in range(num_days+5)] # go back 5 days from the earliest state
    strings_of_interest = [string_from_date(d) for d in dates_of_interest]

    regex_date_matcher = re.compile(r"^.*_(\d{6})_.*\.\w{3}")
    filenames_of_interest = []

    for name in aws_helper.get_list_of_files_s3('leolabs-calibration-sources-test', 'ilrs/'+str(norad_id)): # at this stage no other bucket is required
        try:
            date_component = regex_date_matcher.match(name).group(1)

            if date_component in strings_of_interest:
                filenames_of_interest.append(name)

        except AttributeError:
            pass

    return filenames_of_interest

def download_truth_files(filenames, truth_directory):
    """Downloads truth files from S3 for the current truth target."""

    def date_from_string(dt_string):
        return datetime(2000+int(dt_string[0:2]), int(dt_string[2:4]), int(dt_string[4:6]))

    regex_date_matcher = re.compile(r"^.*_(\d{6})_.*\.\w{3}")
    regex_file_matcher = re.compile(r"^.*/.*/(.*)$")

    num_new_files = 0
    for name in filenames:
        filename = regex_file_matcher.match(name).group(1)

        if not os.path.isfile(truth_directory + '/' + filename):
            aws_helper.download_s3('leolabs-calibration-sources-test', name, truth_directory + '/' + filename)
            num_new_files += 1

    print('info', 'Syncing ILRS truth data from S3 ({} files downloaded)'.format(num_new_files))

def dwld_data_for_target(leolabs_id,epoch_date,num_days):
    """Downloads data for a particular target."""
    norad_id = id_data(leolabs_id)["norad_id"] # look up norad id of target
    trth_dir = set_up_truth_directory_for_target(leolabs_id) # create directory for target
    trth_flnms = get_truth_file_list(epoch_date, norad_id, num_days) # collect all filenames to be downloaded
    download_truth_files(trth_flnms,trth_dir) # download all files
        
def dwld_data_for_all_targets(target_list,epoch,num_days):
    """Downloads data for all targets."""
    for target in target_list:
        dwld_data_for_target(target,epoch,num_days)
        
def collect_all_states(ILRS_target_list, epoch, dates_back_from_epoch):
    """Collects all states for a list of ILRS targets and a specified date range going back from the epoch."""
    tot_state_arr = np.empty(3,dtype='<U32')
    
    for target in ILRS_target_list:
        state_arr = states_available_x_days_from_epoch(target,epoch,dates_back_from_epoch)
        if (state_arr is not None):
            tot_state_arr = np.vstack((tot_state_arr,state_arr.astype('<U32')))
        else:
            pass
        
    return tot_state_arr[1:]

def state_error(object_id,state_id,epoch,timestep = 150, plotting = False):
    """Creates a truth object from a state and runs truth analysis on it, returning the errors."""
    # Initializing object
    id_data_TO = id_data(object_id) # Id data of a truth object
    
    start_time_str, end_time_str = propagation_dates_from_epoch(epoch)
    
    
    propagations_ST = propagation_of_state(object_id,state_id,start_time_str,end_time_str,timestep) # propagate the state and collect the propagations
    propagations_ST_list = propagations_list(propagations_ST) # put the propagations in a container
    ric_covariances_ST = RIC_covariance_of_propagations(object_id,state_id,start_time_str,end_time_str,timestep) # find the covariances of the propagations in the RIC frame
    ric_covariances_ST_list = RIC_Covariances_list(ric_covariances_ST) # put the RIC covariances in a container
    TO = ta.TruthAnalysis(id_data_TO,propagations_ST_list,ric_covariances_ST_list) # Initialize a Truth Object
    
    try: # handling the exception that there are no ILRS truth files to download from S3
        norm_errors_dict, dist_list = TO.ilrs_truth_analysis() # Run Truth Analysis on the TO
        epoch_Offset = extract_epochOffset(norm_errors_dict) # parse epoch Offset
        r_err = extract_norm_error(norm_errors_dict,0) # parse position errors
        i_err = extract_norm_error(norm_errors_dict,1)
        c_err = extract_norm_error(norm_errors_dict,2)

        if plotting:
            plt.figure(figsize=(8,8))
            plt.title(f"RIC Position Error for {object_id} @ epoch {epoch}")
            plt.plot(epoch_Offset,r_err,'g',label='radial')
            plt.plot(epoch_Offset,i_err,'r',label='in-track')
            plt.plot(epoch_Offset,c_err,'b',label='cross-track')
            plt.legend(loc='upper right')
            plt.xlabel('Seconds from estimation epoch')
            plt.ylabel('Error')
            plt.show()
        return epoch_Offset, r_err, i_err, c_err
    except ValueError:
        return None, None, None, None
    
def collections_of_truth_state_errors(ILRS_target_list, epoch, days_from_epoch):
    """Collects all errors for a given list of ILRS targets and dates and returns them in the form of distributions."""
    
    state_array = collect_all_states(ILRS_target_list, epoch, days_from_epoch) # collecting all states for the specified date range
    
    num_of_states = state_array.shape[0]
    
    print("Total number of states = ", num_of_states)

    r_err_list = []
    i_err_list = []
    c_err_list = []
    Ep_Offset_list = []

    counter = 0
    for i in range(num_of_states): # perform truth analysis on each state and store the errors
        obj_id = state_array[i][0]
        state_id = state_array[i][1]
        timestamp = state_array[i][2]

        epoch_Offset, r_err, i_err, c_err = state_error(obj_id,state_id,timestamp)
        counter += 1
        print(f"state {counter}/{num_of_states} done!")
        
        if (r_err is not None):
            Ep_Offset_list.append(epoch_Offset)
            r_err_list.append(r_err)
            i_err_list.append(i_err)
            c_err_list.append(c_err)

        else: 
            pass   
        # convert the lists of errors with len = number_of_time_steps to lists of lists of the same length 
        # but each entry is a list with all the errors of that time step.
        r_err_collection = list(zip_longest(*r_err_list)) 
        i_err_collection = list(zip_longest(*i_err_list))
        c_err_collection = list(zip_longest(*c_err_list))
    
    print("Epoch_Offset_length =",len(Ep_Offset_list))
    print("r_err_length = ",len(r_err_collection))
    print("r_err[0]_length = ",len(r_err_collection[0]))
   
    return Ep_Offset_list[0], r_err_collection, i_err_collection, c_err_collection

# Initialize orekit
orekit_vm = orekit.initVM()
setup_orekit_curdir("/Users/gkeramidas/Projects/learning/leolabs-config-data-dynamic/")

# Initialize aws
api_client, aws_helper = aws_init()

