from datetime import datetime
from dateutil.parser import parse
import os
from typing import Dict, Any, List
import json
import sys
from pathlib import Path

leo_backend_od_path = Path("/Users/gkeramidas/Projects/od-master/leo-backend-od")
sys.path.append(str(leo_backend_od_path))

import requests
import pandas as pd
from pandas import Series
import math
import numpy as np
import asyncio
import aiohttp
import ast

import orekit
from orekit.pyhelpers import setup_orekit_curdir
from org.orekit.bodies import OneAxisEllipsoid, GeodeticPoint
from org.orekit.frames import FramesFactory, TopocentricFrame
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants, IERSConventions

import auth
import radars
from odlib.od_utils.frame_conversion import (
    eci_to_rtn_rotation_matrix,
    eci_covariance_to_rtn,
)


def leolabs_id(row: Series) -> str:
    """Function that takes a dataframe row and returns the leolabs id as a string"""
    return "L" + str(int(row["target_id"]))


def state_id(row: Series) -> str:
    """Function that takes a dataframe row and returns the state id as a string"""
    return str(int(row["state_id"]))


def prop_timestamp(row: Series) -> str:
    """Function that takes a dataframe row and returns the fit_epoch as a string"""
    datetime_obj = parse(row["fit_epoch"])
    return datetime_obj.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def prop_url(row: Series) -> str:
    """Function that takes a dataframe row and returns the propagation url"""
    leo_id = leolabs_id(row)
    state = state_id(row)
    ts = prop_timestamp(row)
    propagation_url = "".join(
        [
            auth.api_url,
            "/catalog/objects/",
            leo_id,
            "/states/",
            state,
            "/propagations?startTime=",
            ts,
            "&endTime=",
            ts,
        ]
    )
    return propagation_url


def handle_propagation_response(eme_response: Dict[str, Any]) -> Series:
    """Function that takes the response from a propagation request and returns the state variables"""
    state_pos = eme_response["propagation"][0]["position"]
    state_vel = eme_response["propagation"][0]["velocity"]
    eme_cov = eme_response["propagation"][0]["covariance"]
    ric_cov = eci_covariance_to_rtn(eme_cov, state_pos, state_vel)

    return pd.Series(
        {
            "X_pr": state_pos[0],
            "Y_pr": state_pos[1],
            "Z_pr": state_pos[2],
            "Vx_pr": state_vel[0],
            "Vy_pr": state_vel[1],
            "Vz_pr": state_vel[2],
            "ric_scC_pr": ric_cov,
            "eme_scC_pr": eme_cov,
        }
    )


def make_columns_from_propagations(row: Series) -> Series:
    """Function that takes a row from the dataframe, calls the propagation API and handles the response
    to create new columns
    """
    eme_url = prop_url(row)

    eme_response = requests.get(eme_url, headers=auth.headers)

    return handle_propagation_response(eme_response.json())


async def make_columns_from_propagations_async(row: Series) -> Series:
    """Function that takes a row from the dataframe, calls the propagation API and handles the response
    to create new columns (asynchronous)
    """
    eme_url = prop_url(row)

    async with aiohttp.ClientSession() as session:
        async with session.get(eme_url, headers=auth.headers) as eme_response:
            eme_response_data = await eme_response.json()

    return handle_propagation_response(eme_response_data)


async def create_propagations_df(df: pd.DataFrame) -> pd.DataFrame:
    tasks = [make_columns_from_propagations_async(row) for _, row in df.iterrows()]
    propagation_columns_series_list = await asyncio.gather(*tasks)
    # 'new_columns_series_list' will contain the Series with new column values for each row

    # Concatenate the original DataFrame with the new columns DataFrame
    # propagation_columns_df = pd.concat(propagation_columns_series_list, axis=0)
    propagation_columns_df = pd.DataFrame(propagation_columns_series_list)
    # result_df = pd.concat([df, new_columns_df], axis=1)

    return propagation_columns_df


def scale_factors_for_row(row: Series, scale_df: pd.DataFrame) -> List:
    """Takes as input a row of the residuals dataframe and the scale factors dataframe and returns the
    RIC scale factors that are created closest to the date of the row's timestamp. Assumes that the scales
    dataframe has the 'created_at' column converted to pandas datetime.
    """
    current_ts = pd.Timestamp(row["timestamp"])
    # creates a new column with the time differences and finds the closest timestamp
    scale_df["time_difference"] = abs(scale_df["created_at"] - current_ts)
    closest_row_index = scale_df["time_difference"].idxmin()
    closest_row = scale_df.loc[closest_row_index]
    # The scaling factors come in the form of a string and we need to split it
    # and evaluate it as a literal to retrieve the list
    scale_data_str = closest_row["scale_data"]
    first_split = scale_data_str.split('"ricScalingFactors":')[1]
    second_split = first_split.split("}")[0]
    scale_list = ast.literal_eval(second_split[1:])
    # drop the unnecessary column for the scale factors dataframe
    scale_df.drop(columns="time_difference", inplace=True)

    return scale_list


def descale_RIC_covariance(ric_cov: np.array, scale_factor_list: List) -> np.array:
    """Takes an RIC covariance matrix and the scale factors and descales the matrix."""
    inverse_scale_factor_list = [1 / x for x in scale_factor_list]
    inverse_scale_factor_matrix = np.diag(np.asarray(inverse_scale_factor_list))

    descaled_ric_cov = np.matmul(
        inverse_scale_factor_matrix, np.matmul(ric_cov, inverse_scale_factor_matrix)
    )
    return descaled_ric_cov


def add_descaled_column(row: Series, scale_df: pd.DataFrame) -> Series:
    """Takes a row and creates a column with a descaled RIC covariance. Works on the concatenated dataframe"""
    scale_list = scale_factors_for_row(row, scale_df)
    dsc_ric_C = descale_RIC_covariance(row["ric_scC_pr"], scale_list)

    return pd.Series({"dsc_ric_C": dsc_ric_C})


def datetime2abs(date: datetime):
    """Converts a datetime object to an Orekit AbsoluteDate"""

    secs = date.second + date.microsecond / 1e6

    abs_date = AbsoluteDate(
        date.year,
        date.month,
        date.day,
        date.hour,
        date.minute,
        secs,
        TimeScalesFactory.getUTC(),
    )
    return abs_date


def radar_coordinates_columns(row: Series) -> Series:
    """Takes a dataframe row and return the coordinates of the radar in the EME2000 frame at the time of measurement"""

    # inputs from the dataframe row
    datetime_epoch = parse(row["fit_epoch"])
    radar_id = int(row["instrument_id"])

    # inputs from the radar dict
    latitude = radars.RADARS[radar_id]["latitude_degrees"]
    longitude = radars.RADARS[radar_id]["longitude_degrees"]
    alt = radars.RADARS[radar_id]["elevation_m"]

    # copy from backend function
    lat = math.radians(latitude)
    long = math.radians(longitude)
    earth_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, False)
    earth_radius = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
    earth_shape = OneAxisEllipsoid(
        earth_radius, Constants.WGS84_EARTH_FLATTENING, earth_frame
    )
    station = GeodeticPoint(lat, long, alt)
    topo = TopocentricFrame(earth_shape, station, "radar")

    inertial_frame = FramesFactory.getEME2000()
    curr_date = datetime2abs(datetime_epoch)

    # Get station position in EME2000
    stn_vector = topo.getPVCoordinates(curr_date, inertial_frame)
    return pd.Series(
        {
            "X_radar": stn_vector.getPosition().getX(),
            "Y_radar": stn_vector.getPosition().getY(),
            "Z_radar": stn_vector.getPosition().getZ(),
            "Vx_radar": stn_vector.getVelocity().getX(),
            "Vy_radar": stn_vector.getVelocity().getY(),
            "Vz_radar": stn_vector.getVelocity().getZ(),
        }
    )


def calculate_residuals(row: Series):
    """Takes a row from the augmented matrix and calculates the residuals from
    propagated and radar positions subtracted from the measured ones
    """
    rad_vec = np.array(
        [
            row["X_radar"],
            row["Y_radar"],
            row["Z_radar"],
            row["Vx_radar"],
            row["Vy_radar"],
            row["Vz_radar"],
        ]
    )
    sat_vec = np.array(
        [
            row["X_pr"],
            row["Y_pr"],
            row["Z_pr"],
            row["Vx_pr"],
            row["Vy_pr"],
            row["Vz_pr"],
        ]
    )

    range_meas = float(row["fit_corrected_range"])
    doppler_meas = float(row["fit_corrected_doppler"])

    x_rel = sat_vec - rad_vec

    # Predicted range
    range_pred = np.linalg.norm(x_rel[0:3])
    # Predicted doppler
    doppler_pred = np.dot(x_rel[0:3], x_rel[3:6]) / range_pred

    range_res = range_meas - range_pred
    doppler_res = doppler_meas - doppler_pred

    return pd.Series(
        {
            "R_res_calc": range_res,
            "RR_res_calc": doppler_res,
            "range_pred": range_pred,
            "doppler_pred": doppler_pred,
            "range_meas": range_meas,
            "doppler_meas": doppler_meas,
        }
    )
