import json
from operator import truth
import os
import requests
import numpy as np
import datetime
from datetime import timedelta
import auth
import truth_analysis
from dateutil.parser import parse as date_parse


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