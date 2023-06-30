import os
import sys
from pathlib import Path

leo_backend_od_path = Path("/Users/gkeramidas/Projects/od-master/leo-backend-od")
sys.path.append(str(leo_backend_od_path))


import re
import datetime
from datetime import timedelta
import statistics
from itertools import zip_longest
import shutil
from typing import Dict, List, Any, Union

import requests
import numpy as np
from dateutil.parser import parse as date_parse
from matplotlib import pyplot as plt
import asyncio

from utilities.aws_helper import AwsHelper
from utilities.api import Api
import auth
import truth_analysis_local as tal

api_client = Api.get_client()
aws_helper = AwsHelper()


def id_data(leo_id: str) -> Dict[str, Any]:
    """ID Data for a LeoLabs object to initialize the truth analysis object."""
    object_url = "".join([auth.api_url, "/catalog/objects/", leo_id])
    object_response = requests.get(object_url, headers=auth.headers)
    leolabs_id = object_response.json()["catalogNumber"]
    norad_id = object_response.json()["noradCatalogNumber"]
    name = object_response.json()["name"]

    id_dict = {}
    id_dict["leolabs_id"] = leolabs_id
    id_dict["norad_id"] = norad_id
    id_dict["object_name"] = name
    return id_dict


def request_response(object_url: str) -> Dict[str, Any]:
    request_response = requests.get(object_url, headers=auth.headers)
    return request_response


async def async_id_data(leo_id: str) -> Dict[str, Any]:
    object_url = "".join([auth.api_url, "/catalog/objects/", leo_id])
    object_response = await asyncio.to_thread(request_response, object_url)
    leolabs_id = object_response.json()["catalogNumber"]
    norad_id = object_response.json()["noradCatalogNumber"]
    name = object_response.json()["name"]

    id_dict = {}
    id_dict["leolabs_id"] = leolabs_id
    id_dict["norad_id"] = norad_id
    id_dict["object_name"] = name
    return id_dict


def propagation_dates_from_epoch(epoch: List[int]) -> tuple[str, str]:
    """Function handed an epoch (as a list) and returning start and end days, 1 day away from epoch"""

    epoch_in_dt = datetime.datetime(*epoch, tzinfo=None)
    start_time = epoch_in_dt - timedelta(
        seconds=86399
    )  # start of propagations is 1 day before epoch
    end_time = epoch_in_dt + timedelta(
        seconds=86401
    )  # end of propagations is 1 day after epoch

    start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    return start_time_str, end_time_str


def states_available_x_days_from_epoch(
    object_id: str, epoch: List[int], num_days: int
) -> Union[np.ndarray, None]:
    """Takes an object id and a date and returns all the states available for this target in a number of days prior to epoch"""

    end_time_str = datetime.datetime(epoch[0], epoch[1], epoch[2]).strftime(
        "%Y-%m-%d"
    )  # end time is the epoch
    start_time = datetime.datetime(epoch[0], epoch[1], epoch[2]) - timedelta(
        days=num_days
    )  # start time is num_days prior to epoch
    start_time_str = start_time.strftime("%Y-%m-%d")

    states_url = "".join(
        [
            auth.api_url,
            "/catalog/objects/",
            object_id,
            "/states?startTime=",
            start_time_str,
            "&endTime=",
            end_time_str,
        ]
    )

    states_response = requests.get(states_url, headers=auth.headers)

    states_list = states_response.json()["states"]

    if len(states_list) > 0:  # handling the exception where there are no states

        states_arr = np.empty(3, dtype="<U32")

        for i in range(len(states_list)):
            states_arr = np.vstack(
                (
                    states_arr,
                    [object_id, states_list[i]["id"], states_list[i]["timestamp"]],
                )
            )

        return states_arr[1:].astype("<U32")
    else:
        return None


def state_data(
    leo_id: str, start_date: List[int], end_date: List[int]
) -> Dict[str, Any]:
    """Pull all the states of a LeoLabs object between certain dates"""

    start_time = datetime.datetime(
        start_date[0], start_date[1], start_date[2]
    ).strftime("%Y-%m-%d")
    end_time = datetime.datetime(end_date[0], end_date[1], end_date[2]).strftime(
        "%Y-%m-%d"
    )

    states_url = "".join(
        [
            auth.api_url,
            "/catalog/objects/",
            leo_id,
            "/states?startTime=",
            start_time,
            "&endTime=",
            end_time,
        ]
    )

    states_response = requests.get(states_url, headers=auth.headers)

    return states_response


def RIC_covariance_of_propagations(
    leo_id: str, state_id: str, start_time: str, end_time: str, timestep: int
) -> List[Dict[str, Any]]:
    """Pull the RIC covariances out of the propagations"""

    prop_with_ric_url = "".join(
        [
            auth.api_url,
            "/catalog/objects/",
            leo_id,
            "/states/",
            state_id,
            "/propagations?startTime=",
            start_time,
            "&endTime=",
            end_time,
            "&timestep=",
            str(timestep),
            "&frame=RIC",
        ]
    )

    prop_with_ric_response = requests.get(prop_with_ric_url, headers=auth.headers)

    return prop_with_ric_response.json()["propagation"]


async def async_RIC_covariance_of_propagations(
    leo_id: str, state_id: str, start_time: str, end_time: str, timestep: int
) -> List[Dict[str, Any]]:
    """Asynchonously pull the RIC covariances out of the propagations"""

    prop_with_ric_url = "".join(
        [
            auth.api_url,
            "/catalog/objects/",
            leo_id,
            "/states/",
            state_id,
            "/propagations?startTime=",
            start_time,
            "&endTime=",
            end_time,
            "&timestep=",
            str(timestep),
            "&frame=RIC",
        ]
    )

    prop_with_ric_response = await asyncio.to_thread(
        request_response, prop_with_ric_url
    )

    return prop_with_ric_response.json()["propagation"]


def propagation_of_state(
    leo_id: str, state_id: str, start_time: str, end_time: str, timestep: int
) -> List[Dict[str, Any]]:
    """Request state propagations between a start date and an end date and with certain timestep."""

    propagation_url = "".join(
        [
            auth.api_url,
            "/catalog/objects/",
            leo_id,
            "/states/",
            state_id,
            "/propagations?startTime=",
            start_time,
            "&endTime=",
            end_time,
            "&timestep=",
            str(timestep),
        ]
    )

    propagation_response = requests.get(propagation_url, headers=auth.headers)

    return propagation_response.json()["propagation"]


async def async_propagation_of_state(
    leo_id: str, state_id: str, start_time: str, end_time: str, timestep: int
) -> Dict[str, Any]:
    """Asynchronous request of state propagations between a start date and an end date and with certain timestep."""

    propagation_url = "".join(
        [
            auth.api_url,
            "/catalog/objects/",
            leo_id,
            "/states/",
            state_id,
            "/propagations?startTime=",
            start_time,
            "&endTime=",
            end_time,
            "&timestep=",
            str(timestep),
        ]
    )

    propagation_response = await asyncio.to_thread(request_response, propagation_url)

    return propagation_response.json()["propagation"]


class PropagationsContainer:
    """Convenience container for propagations"""

    def __init__(
        self,
        timestamp: datetime.datetime,
        position: List[float],
        velocity: List[float],
        covariance: List[float],
    ):
        self.timestamp = timestamp
        self.position = position
        self.velocity = velocity
        self.covariance = covariance


class FullRicCovariancesContainer:
    """Convenience container for Full Ric propagations"""

    def __init__(
        self,
        timestamp: datetime.datetime,
        position: List[float],
        velocity: List[float],
        covariance: List[float],
    ):
        self.timestamp = timestamp
        self.position = position
        self.velocity = velocity
        self.covariance = covariance


class RicCovariancesContainer:
    """Convenience container for RIC covariances"""

    def __init__(self, covariance: List[float]):
        self.covariance = covariance


def Ric_propagation_dict_to_container(
    ric_prop_json: Dict[str, Any]
) -> RicCovariancesContainer:
    """Takes a single RIC propagation json dictionary and returns a RicCovariancesContainer object."""

    covariance = ric_prop_json["covariance"]

    return RicCovariancesContainer(covariance)


def Full_Ric_propagation_dict_to_container(
    ric_prop_json: Dict[str, Any]
) -> FullRicCovariancesContainer:
    """Takes a single Full RIC propagation json dictionary and returns a FullRicCovariancesContainer object."""

    timestamp = date_parse(ric_prop_json["timestamp"], ignoretz=True)
    position = ric_prop_json["position"]
    velocity = ric_prop_json["velocity"]
    covariance = ric_prop_json["covariance"]

    return FullRicCovariancesContainer(timestamp, position, velocity, covariance)


def propagation_dict_to_container(prop_json: Dict[str, Any]) -> PropagationsContainer:
    """Takes a single propagation list and returns a PropagationsContainer object."""

    timestamp = date_parse(prop_json["timestamp"], ignoretz=True)
    position = prop_json["position"]
    velocity = prop_json["velocity"]
    covariance = prop_json["covariance"]

    return PropagationsContainer(timestamp, position, velocity, covariance)


def propagations_list(
    propagations_json_list: List[Dict[str, Any]]
) -> List[PropagationsContainer]:
    """Takes a list of propagations jsons and returns a list with PropagationContainer objects."""

    prop_list = list(map(propagation_dict_to_container, propagations_json_list))

    return prop_list


def RIC_Covariances_list(
    ric_json: List[Dict[str, Any]]
) -> List[RicCovariancesContainer]:
    """Takes a list of RIC propagations json dicts and returns a list of RicCovariancesContainer object."""

    ric_list = list(
        map(
            lambda obj: obj.covariance, map(Ric_propagation_dict_to_container, ric_json)
        )
    )

    return ric_list


def Full_RIC_Covariances_list(
    ric_json: List[Dict[str, Any]]
) -> List[FullRicCovariancesContainer]:
    """Takes a list of Full RIC propagations json dicts and returns a list of FullRicCovariancesContainer objects."""

    ric_list = list(map(Full_Ric_propagation_dict_to_container, ric_json))

    return ric_list


def extract_epochOffset(norm_errors_dicts: List[Dict[str, Any]]) -> List[float]:
    """Takes a list of dictionaries with normalized errors, extracts the epoch Offsets and returns them into a list"""

    epoch_Offset_list = list(map(lambda obj: obj["epochOffset"], norm_errors_dicts))

    return epoch_Offset_list


def extract_norm_error(
    norm_errors_dicts: List[Dict[str, Any]], coord_ind: int
) -> List[float]:
    """Takes a list of dictionaries with normalized errors and extracts any of the errors, based on the coord_ind"""

    error_list = list(map(lambda obj: obj["vals"][coord_ind], norm_errors_dicts))

    return error_list


def z_score(d: float, MAD: float) -> float:

    return 0.6745 * d / MAD


def cull_outliers(arr: List[float]) -> List[float]:
    """Function for culling outliers based on their mean squared distance from the median."""
    threshold = 3.0

    arr = np.array(arr)

    # Strip nans
    arr = arr[~np.isnan(arr)]

    diffs = np.sqrt((arr - np.median(arr)) ** 2)

    MAD = np.median(diffs)

    return [y for x, y in zip(diffs, arr) if z_score(x, MAD) < threshold]


def extract_std_from_error_distributions(err_collection) -> List[float]:
    """Takes a collection of error distributions and returns the standard deviation at each time step."""

    stdevs = list(map(lambda x: statistics.pstdev(cull_outliers(x)), err_collection))

    return stdevs


def set_up_truth_directory_for_target(
    leolabs_id: Any, absolute_path_to_truth_dir: Path
) -> str:
    """Prepares a local directory for storing ILRS truth files for particular ILRS target."""
    try:
        shutil.rmtree(
            str(absolute_path_to_truth_dir) + "/truth/" + str(leolabs_id)
        )  # Remove target's directory if it already exists
    except FileNotFoundError:
        pass

    base_truth_directory = str(absolute_path_to_truth_dir) + "/truth/"
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


def get_truth_file_list(
    epoch_date: List[int], norad_id: int, num_days: int
) -> List[str]:
    """Builds list of S3 truth files for the current object that match the date range of interest."""

    epoch_date_in_dt = datetime.datetime(
        epoch_date[0], epoch_date[1], epoch_date[2]
    )  # we need this date in datetime

    def string_from_date(dt):
        return str(dt.year)[-2:] + "%02d" % (dt.month,) + "%02d" % (dt.day,)

    dates_of_interest = [
        (epoch_date_in_dt - timedelta(days=i)) for i in range(num_days + 5)
    ]  # go back 5 days from the earliest state
    strings_of_interest = [string_from_date(d) for d in dates_of_interest]

    regex_date_matcher = re.compile(r"^.*_(\d{6})_.*\.\w{3}")
    filenames_of_interest = []

    for name in aws_helper.get_list_of_files_s3(
        "leolabs-calibration-sources", "ilrs/" + str(norad_id)
    ):  # at this stage no other bucket is required
        try:
            date_component = regex_date_matcher.match(name).group(1)

            if date_component in strings_of_interest:
                filenames_of_interest.append(name)

        except AttributeError:
            pass

    return filenames_of_interest


def download_truth_files(filenames: List[str], truth_directory: str):
    """Downloads truth files from S3 for the current truth target."""

    def date_from_string(dt_string):
        return datetime(
            2000 + int(dt_string[0:2]), int(dt_string[2:4]), int(dt_string[4:6])
        )

    regex_date_matcher = re.compile(r"^.*_(\d{6})_.*\.\w{3}")
    regex_file_matcher = re.compile(r"^.*/.*/(.*)$")

    num_new_files = 0
    for name in filenames:
        filename = regex_file_matcher.match(name).group(1)

        if not os.path.isfile(truth_directory + "/" + filename):
            # aws_helper.download_s3('leolabs-calibration-sources-test', name, truth_directory + '/' + filename)
            aws_helper.download_s3(
                "leolabs-calibration-sources", name, truth_directory + "/" + filename
            )
            num_new_files += 1

    print(
        "info",
        "Syncing ILRS truth data from S3 ({} files downloaded)".format(num_new_files),
    )


def dwld_data_for_target(
    leolabs_id: str,
    epoch_date: List[int],
    num_days: int,
    absolute_path_to_truth_dir: Path,
):
    """Downloads data for a particular target."""
    norad_id = id_data(leolabs_id)["norad_id"]  # look up norad id of target
    trth_dir = set_up_truth_directory_for_target(
        leolabs_id, absolute_path_to_truth_dir
    )  # create directory for target
    trth_flnms = get_truth_file_list(
        epoch_date, norad_id, num_days
    )  # collect all filenames to be downloaded
    download_truth_files(trth_flnms, trth_dir)  # download all files


def dwld_data_for_all_targets(target_list: List[str], epoch: List[int], num_days: int):
    """Downloads data for all targets."""
    for target in target_list:
        dwld_data_for_target(target, epoch, num_days)


def collect_all_states(
    ILRS_target_list: List[str], epoch: List[int], dates_back_from_epoch: int
) -> np.ndarray:
    """Collects all states for a list of ILRS targets and a specified date range going back from the epoch."""
    tot_state_arr = np.empty(3, dtype="<U32")

    for target in ILRS_target_list:
        state_arr = states_available_x_days_from_epoch(
            target, epoch, dates_back_from_epoch
        )
        if state_arr is not None:
            tot_state_arr = np.vstack((tot_state_arr, state_arr.astype("<U32")))
        else:
            pass

    return tot_state_arr[1:]


def state_error(
    object_id: str, state_id: str, epoch: List[int], timestep=150, plotting=False
) -> Union[tuple[float, float, float, float], tuple[None, None, None, None]]:
    """Creates a truth object from a state and runs truth analysis on it, returning the errors."""
    # Initializing object
    id_data_TO = id_data(object_id)  # Id data of a truth object

    start_time_str, end_time_str = propagation_dates_from_epoch(epoch)

    propagations_ST = propagation_of_state(
        object_id, state_id, start_time_str, end_time_str, timestep
    )  # propagate the state and collect the propagations
    propagations_ST_list = propagations_list(
        propagations_ST
    )  # put the propagations in a container
    ric_covariances_ST = RIC_covariance_of_propagations(
        object_id, state_id, start_time_str, end_time_str, timestep
    )  # find the covariances of the propagations in the RIC frame
    ric_covariances_ST_list = RIC_Covariances_list(
        ric_covariances_ST
    )  # put the RIC covariances in a container
    TO = tal.TruthAnalysis(
        id_data_TO, propagations_ST_list, ric_covariances_ST_list
    )  # Initialize a Truth Object

    try:  # handling the exception that there are no ILRS truth files to download from S3
        (
            norm_errors_dict,
            dist_list,
        ) = TO.ilrs_truth_analysis()  # Run Truth Analysis on the TO
        epoch_Offset = extract_epochOffset(norm_errors_dict)  # parse epoch Offset
        r_err = extract_norm_error(norm_errors_dict, 0)  # parse position errors
        i_err = extract_norm_error(norm_errors_dict, 1)
        c_err = extract_norm_error(norm_errors_dict, 2)

        if plotting:
            plt.figure(figsize=(8, 8))
            plt.title(f"RIC Position Error for {object_id} @ epoch {epoch}")
            plt.plot(epoch_Offset, r_err, "g", label="radial")
            plt.plot(epoch_Offset, i_err, "r", label="in-track")
            plt.plot(epoch_Offset, c_err, "b", label="cross-track")
            plt.legend(loc="upper right")
            plt.xlabel("Seconds from estimation epoch")
            plt.ylabel("Error")
            plt.show()
        return epoch_Offset, r_err, i_err, c_err
    except ValueError:
        return None, None, None, None


async def async_state_requests(
    object_id: str, state_id: str, epoch: List[int], timestep=150
) -> tal.TruthAnalysis:
    """Group all API requests needed for state error estimation."""
    # Initializing object
    id_data_TO = await async_id_data(object_id)  # Id data of a truth object

    start_time_str, end_time_str = propagation_dates_from_epoch(epoch)

    propagations_ST = await async_propagation_of_state(
        object_id, state_id, start_time_str, end_time_str, timestep
    )  # propagate the state and collect the propagations
    propagations_ST_list = propagations_list(
        propagations_ST
    )  # put the propagations in a container
    ric_covariances_ST = await async_RIC_covariance_of_propagations(
        object_id, state_id, start_time_str, end_time_str, timestep
    )  # find the covariances of the propagations in the RIC frame
    ric_covariances_ST_list = RIC_Covariances_list(
        ric_covariances_ST
    )  # put the RIC covariances in a container
    TO = tal.TruthAnalysis(
        id_data_TO, propagations_ST_list, ric_covariances_ST_list
    )  # Initialize a Truth Object
    return TO


def truth_analysis_errors(
    truth_object: tal.TruthAnalysis,
) -> Union[
    tuple[float, float, float, float, float, float, float],
    tuple[None, None, None, None, None, None, None],
]:
    """Second part of original state_error function. Designed to not contain API requests."""
    try:  # handling the exception that there are no ILRS truth files to download from S3
        (
            norm_errors_dict,
            dist_list,
        ) = truth_object.ilrs_truth_analysis()  # Run Truth Analysis on the TO
        epoch_Offset = extract_epochOffset(norm_errors_dict)  # parse epoch Offset
        r_err = extract_norm_error(norm_errors_dict, 0)  # parse position errors
        i_err = extract_norm_error(norm_errors_dict, 1)
        c_err = extract_norm_error(norm_errors_dict, 2)
        Vr_err = extract_norm_error(norm_errors_dict, 3)  # parse position errors
        Vi_err = extract_norm_error(norm_errors_dict, 4)
        Vc_err = extract_norm_error(norm_errors_dict, 5)

        return epoch_Offset, r_err, i_err, c_err, Vr_err, Vi_err, Vc_err
    except ValueError:
        return None, None, None, None, None, None, None


def collections_of_truth_state_errors(
    ILRS_target_list: List[str], epoch: List[int], days_from_epoch: int
) -> tuple[float, float, float, float, float, float, float]:
    """Collects all errors for a given list of ILRS targets and dates and returns them in the form of distributions."""

    state_array = collect_all_states(
        ILRS_target_list, epoch, days_from_epoch
    )  # collecting all states for the specified date range

    num_of_states = state_array.shape[0]

    print("Total number of states = ", num_of_states)

    r_err_list = []
    i_err_list = []
    c_err_list = []
    Vr_err_list = []
    Vi_err_list = []
    Vc_err_list = []
    Ep_Offset_list = []

    counter = 0
    for i in range(
        num_of_states
    ):  # perform truth analysis on each state and store the errors
        obj_id = state_array[i][0]
        state_id = state_array[i][1]
        timestamp = state_array[i][2]

        epoch_Offset, r_err, i_err, c_err, Vr_err, Vi_err, Vc_err = state_error(
            obj_id, state_id, timestamp
        )
        counter += 1
        print(f"state {counter}/{num_of_states} done!")

        if r_err is not None:
            Ep_Offset_list.append(epoch_Offset)
            r_err_list.append(r_err)
            i_err_list.append(i_err)
            c_err_list.append(c_err)
            Vr_err_list.append(Vr_err)
            Vi_err_list.append(Vi_err)
            Vc_err_list.append(Vc_err)

        else:
            pass
        # convert the lists of errors with len = number_of_time_steps to lists of lists of the same length
        # but each entry is a list with all the errors of that time step.
        r_err_collection = list(zip_longest(*r_err_list))
        i_err_collection = list(zip_longest(*i_err_list))
        c_err_collection = list(zip_longest(*c_err_list))
        Vr_err_collection = list(zip_longest(*Vr_err_list))
        Vi_err_collection = list(zip_longest(*Vi_err_list))
        Vc_err_collection = list(zip_longest(*Vc_err_list))

    print("Epoch_Offset_length =", len(Ep_Offset_list))
    print("r_err_length = ", len(r_err_collection))
    print("r_err[0]_length = ", len(r_err_collection[0]))

    return (
        Ep_Offset_list[0],
        r_err_collection,
        i_err_collection,
        c_err_collection,
        Vr_err_collection,
        Vi_err_collection,
        Vc_err_collection,
    )
