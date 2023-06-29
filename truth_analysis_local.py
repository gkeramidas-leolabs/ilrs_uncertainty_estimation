"""
This module contains a class for comparing LeoLabs states to known sources
of truth.
"""
import sys
from pathlib import Path

leo_backend_od_path = Path("/Users/gkeramidas/Projects/od-master/leo-backend-od")
sys.path.append(str(leo_backend_od_path))

import re
import os
import time
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
import orekit
from orekit.pyhelpers import setup_orekit_curdir

from odlib.ilrs import TruthEphemerisManager
from odlib.od_utils.frame_conversion import eci_to_rtn_rotation_matrix

from utilities.aws_helper import AwsHelper
from utilities.api import Api
from utilities.od_logging import OptionalLog

import error_retrieval as er

# Initialize orekit
orekit_vm = orekit.initVM()
setup_orekit_curdir(
    "/Users/gkeramidas/Projects/ilrs_uncertainty_estimation/leolabs-config-data-dynamic/"
)

truth_path = Path("/Users/gkeramidas/Projects/ilrs_uncertainty_estimation/truth")


class TruthAnalysis(OptionalLog):
    """
    Class that provides truth analysis on objects that are sources of truth.
    """

    def __init__(
        self,
        id_data: Dict[str, Any],
        propagation: List["truth_request.PropagationsContainer"],
        ric_covariances: List["truth_request.RicCovariancesContainer"],
        logger=None,
        upload_truth_plot=False,
        flush_truth_files=True,
    ):
        # TODO: Create truth target file to eliminate hard coding here
        # More ilrs targets: L7145 L12821 L15200 L5520 L13257 L225 L1768 L6888

        self.leolabs_id = id_data["leolabs_id"]
        self.norad_id = id_data["norad_id"]
        self.name = id_data["object_name"]
        self._logger = logger

        if self.leolabs_id in [
            "L5011",
            "L3059",
            "L335",
            "L2486",
            "L4884",
            "L1471",
            "L5429",
            "L3972",
            "L3969",
            "L2669",
            "L2682",
            "L3226",
        ]:

            self.is_truth_object = True
        else:
            self.is_truth_object = False

        # Initialize API connection object
        self._api = Api.get_client(logger=logger)

        # Initialize AWS helper functions
        self._aws = AwsHelper(logger=logger)

        self._upload_truth_plot = upload_truth_plot
        self._flush_truth_files = flush_truth_files

        self._first_date = propagation[0].timestamp
        self._epoch_date = self._first_date + timedelta(days=1)
        self._timestep = (
            propagation[1].timestamp - propagation[0].timestamp
        ).total_seconds()
        self._one_day_in_time_steps = int((24.0 * 60.0 * 60.0) / self._timestep)

        self._propagation = propagation
        self._ric_covariances = ric_covariances

        env_name = os.environ.get("ENVIRONMENT_NAME")
        if env_name == "stage":
            self.ilrs_ephem_bucket = "leolabs-calibration-sources-stage"
        elif env_name == "test":
            self.ilrs_ephem_bucket = "leolabs-calibration-sources-test"
        elif env_name == "prd":
            self.ilrs_ephem_bucket = "leolabs-calibration-sources"
        else:
            self.ilrs_ephem_bucket = "leolabs-calibration-sources-test"

    def ilrs_truth_analysis(self) -> tuple[Dict[str, Any], List[float]]:
        """
        Calculates position and velocity error in RIC frame between calculated
        target state and ILRS truth source. Returns mahalanobis distances for
        these errors and the mangitude of the position error
        """

        # Generate ephemeris data
        time_list = []
        dist_list = []
        spd_list = []

        normalized_pos_error_r = []
        normalized_pos_error_i = []
        normalized_pos_error_c = []

        normalized_vel_error_r = []
        normalized_vel_error_i = []
        normalized_vel_error_c = []

        epoch_date_unix = (
            self._epoch_date - datetime(1970, 1, 1, 0, 0, 0)
        ).total_seconds()
        curr_date_unix = self._first_date.replace(
            tzinfo=timezone.utc
        ).timestamp()  # Initial code

        # ilrs_err_vec = er.get_ilrs_uncertainty(self.leolabs_id)
        ilrs_err_vec = [0, 0, 0, 0, 0, 0]
        ilrs_pos_err = ilrs_err_vec[0:3]
        ilrs_vel_err = ilrs_err_vec[3:]

        eph = self._get_truth_ephemeris_manager()

        time_index = 2 * self._one_day_in_time_steps + 1

        for state, ric_cov in zip(
            self._propagation[:time_index], self._ric_covariances[:time_index]
        ):
            t = curr_date_unix - epoch_date_unix

            position = state.position
            velocity = state.velocity

            # First position
            eme_dist_vec = np.array(
                eph.position_at_unix_time(curr_date_unix)
            ) - np.array(position)
            ric_dist_vec = np.matmul(
                eci_to_rtn_rotation_matrix(position, velocity), eme_dist_vec
            )
            ilrs_ric_pos_err = np.matmul(
                eci_to_rtn_rotation_matrix(position, velocity), ilrs_pos_err
            )

            dist = np.linalg.norm(ric_dist_vec)
            dist_list.append(dist)

            ilrs_ric_err0 = ilrs_ric_pos_err[0]
            ilrs_ric_err1 = ilrs_ric_pos_err[1]
            ilrs_ric_err2 = ilrs_ric_pos_err[2]

            normalized_pos_error_r.append(
                ric_dist_vec[0] / (np.sqrt(ric_cov[0][0] + ilrs_ric_err0**2))
            )
            normalized_pos_error_i.append(
                ric_dist_vec[1] / (np.sqrt(ric_cov[1][1] + ilrs_ric_err1**2))
            )
            normalized_pos_error_c.append(
                ric_dist_vec[2] / (np.sqrt(ric_cov[2][2] + ilrs_ric_err2**2))
            )

            # Then velocity
            eme_spd_vec = np.array(
                eph.derived_velocity_at_unix_time(curr_date_unix)
            ) - np.array(velocity)
            ric_spd_vec = np.matmul(
                eci_to_rtn_rotation_matrix(position, velocity), eme_spd_vec
            )
            ilrs_ric_spd_err = np.matmul(
                eci_to_rtn_rotation_matrix(position, velocity), ilrs_vel_err
            )

            spd = np.linalg.norm(ric_spd_vec)
            spd_list.append(spd)

            normalized_vel_error_r.append(
                ric_spd_vec[0] / (np.sqrt(ric_cov[3][3] + ilrs_ric_spd_err[0] ** 2))
            )
            normalized_vel_error_i.append(
                ric_spd_vec[1] / (np.sqrt(ric_cov[4][4] + ilrs_ric_spd_err[1] ** 2))
            )
            normalized_vel_error_c.append(
                ric_spd_vec[2] / (np.sqrt(ric_cov[5][5] + ilrs_ric_spd_err[2] ** 2))
            )

            time_list.append(t)

            curr_date_unix += self._timestep

        timestamp_id = (
            "".join(
                [
                    str(x).zfill(2)
                    for x in [
                        self._epoch_date.year,
                        self._epoch_date.month,
                        self._epoch_date.day,
                    ]
                ]
            )
            + "_"
            + "".join(
                [
                    str(x).zfill(2)
                    for x in [
                        self._epoch_date.hour,
                        self._epoch_date.minute,
                        self._epoch_date.second,
                    ]
                ]
            )
            + "_"
            + str(self._epoch_date.microsecond)[:3].zfill(3)
        )

        normalized_error_vectors = zip(
            normalized_pos_error_r,
            normalized_pos_error_i,
            normalized_pos_error_c,
            normalized_vel_error_r,
            normalized_vel_error_i,
            normalized_vel_error_c,
        )

        normalized_errors = [
            {"epochOffset": t, "vals": val}
            for t, val in zip(time_list, normalized_error_vectors)
        ]

        # Upload plot to S3
        if self._upload_truth_plot:
            self._plot_and_upload(eph, time_list, dist_list, timestamp_id)

        return normalized_errors, dist_list

    def _get_truth_ephemeris_manager(self, num_days=5) -> TruthEphemerisManager:
        """
        Generates a TruthEphemerisManager object from truth files.
        """
        files = self._get_relevant_truth_files(num_days)
        # print("Epoch timestamp:", self._epoch_date)
        eph_compose_start_time = time.time()
        eph = TruthEphemerisManager(files)
        eph_compose_duration = time.time() - eph_compose_start_time
        self._log(
            "info",
            "Truth data reader assembled ({} seconds elapsed)".format(
                eph_compose_duration
            ),
        )

        return eph

    def _get_relevant_truth_files(self, num_days=5, truth_path=truth_path) -> List[str]:
        """Look up in the truth directory of the target and fetch the files from the relevant dates to initialize the TruthEphemerisManager"""

        target_truth_directory = str(truth_path) + "/" + str(self.leolabs_id) + "/"

        def string_from_date(dt):
            return str(dt.year)[-2:] + "%02d" % (dt.month,) + "%02d" % (dt.day,)

        dates_of_interest = [
            (self._epoch_date - timedelta(days=i)) for i in range(num_days)
        ]
        strings_of_interest = [string_from_date(d) for d in dates_of_interest]

        regex_date_matcher = re.compile(r"^.*_(\d{6})_.*\.\w{3}")
        filenames_of_interest = []

        for name in os.listdir(target_truth_directory):
            try:
                date_component = regex_date_matcher.match(name).group(1)

                if date_component in strings_of_interest:
                    filenames_of_interest.append(target_truth_directory + name)

            except AttributeError:
                pass

        return filenames_of_interest

    def _plot_and_upload(self, eph, time_list, dist_list, timestamp_id):
        """
        Plots truth distances vs time and uploads plot to S3.
        """

        plot_len = self._one_day_in_time_steps
        plt.plot(time_list[: plot_len + 1], dist_list[: plot_len + 1], "b-")

        plt.xlabel("Epoch: " + self._epoch_date.isoformat())
        plt.ylabel("Distance residual (m)")
        plt.title(self.name + " (" + eph.source + ")")

        image_buffer = BytesIO()
        plt.savefig(image_buffer)
        image_buffer.seek(0)

        env_name = os.environ.get("ENVIRONMENT_NAME")
        if env_name == "stage":
            bucket_name = "leolabs-od-truth-comparison-stage"
        elif env_name == "test":
            bucket_name = "leolabs-od-truth-comparison-test"
        elif env_name == "prd":
            bucket_name = "leolabs-od-truth-comparison"
        else:
            bucket_name = "leolabs-od-truth-comparison-test"

        try:
            use_local_test = bool(int(os.environ["LL_LOCAL_TESTING"]))
        except KeyError:
            use_local_test = False

        plot_filename_for_s3 = self.leolabs_id + "_" + timestamp_id + ".png"
        # self._aws.upload_s3(bucket_name, 'plots/' + plot_filename_for_s3, image_buffer)
