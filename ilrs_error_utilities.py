import os
from datetime import datetime
import statistics
from typing import List, Union

import numpy as np
import scipy.stats
import scipy.optimize
import orekit
from orekit.pyhelpers import setup_orekit_curdir

from odlib.ilrs import TruthEphemerisManager
from odlib.od_utils.frame_conversion import eci_to_rtn_rotation_matrix


# Initialize orekit
orekit_vm = orekit.initVM()
setup_orekit_curdir(
    "/Users/gkeramidas/Projects/ilrs_uncertainty_estimation/leolabs-config-data-dynamic/"
)


class tephem:
    """
    Class that takes truth ephemeris objects with their metadata.
    """

    def __init__(
        self,
        year: int,
        month: int,
        day: int,
        ephem: TruthEphemerisManager,
        name: str,
        ftype: str,
    ):
        self.year = year
        self.month = month
        self.day = day
        self.ephem = ephem
        self.name = name
        self.ftype = ftype


def calculate_epoch_unix(year: int, month: int, day: int) -> float:
    """Calculates epoch unix from the date of epoch."""

    epoch_dt = datetime(year, month, day)
    epoch_unix = (epoch_dt - datetime(1970, 1, 1, 0, 0, 0)).total_seconds()
    return epoch_unix


def next_day(year: int, month: int, day: int) -> tuple[int, int, int]:
    """Takes one day and returns the next one. No leap years.

    TODO: ADD leap years
    """

    day_list = [int(i) for i in np.linspace(1, 31, 31)]
    month_list = [int(i) for i in np.linspace(1, 12, 12)]

    if month in [1, 3, 5, 7, 8, 10, 12]:
        dlist = day_list
    if month == 2:
        dlist = day_list[0:28]
    if month in [4, 6, 9, 11]:
        dlist = day_list[0:30]

    next_day = np.roll(dlist, -1)[day - 1]

    if next_day == 1:
        next_month = np.roll(month_list, -1)[month - 1]
    else:
        next_month = month

    if next_day == 1 and next_month == 1:
        next_year = year + 1
    else:
        next_year = year

    return next_year, next_month, next_day


def previous_day(year: int, month: int, day: int) -> tuple[int, int, int]:
    """Takes an epoch and returns the previous day. No leap years.

    TODO: Add leap years
    """

    day_list = [int(i) for i in np.linspace(1, 31, 31)]
    month_list = [int(i) for i in np.linspace(1, 12, 12)]

    if day == 1:
        prev_month = np.roll(month_list, +1)[month - 1]
    else:
        prev_month = month

    if prev_month in [1, 3, 5, 7, 8, 10, 12]:
        dlist = day_list
    if prev_month == 2:
        dlist = day_list[0:28]
    if prev_month in [4, 6, 9, 11]:
        dlist = day_list[0:30]

    prev_day = np.roll(dlist, +1)[day - 1]
    prev_year = year

    if day == 1 and month == 1:
        prev_year = year - 1
    else:
        prev_year = year

    return prev_year, prev_month, prev_day


def base_date(epoch: List[int], num_days: int) -> tuple[int, int, int]:
    """Takes epoch date and looks as many days back to find the base date."""

    ep_year = epoch[0]
    ep_month = epoch[1]
    ep_day = epoch[2]

    for i in range(num_days):
        ep_year, ep_month, ep_day = previous_day(ep_year, ep_month, ep_day)

    return ep_year, ep_month, ep_day


def find_ephem(
    ephem_list: List[TruthEphemerisManager], year: int, month: int, day: int, ftype: str
) -> Union(TruthEphemerisManager, None):
    """Finds ephemeris object based on epoch and file type."""

    year = int("".join(list(str(year)))[2:])

    chosen_ephems = filter(
        lambda eph: eph.year == year
        and eph.month == month
        and eph.day == day
        and eph.ftype == ftype,
        ephem_list,
    )

    return next(chosen_ephems, None)


def locate_ephemerides_for_date(
    ephem_list: List[TruthEphemerisManager],
    year: int,
    month: int,
    day: int,
    prov_list: List[str],
) -> List[TruthEphemerisManager]:
    """Finds all ephemerides of a certain date."""

    candidate_ephem = list(
        filter(
            lambda eph: eph.year == year
            and eph.month == month
            and eph.day == day
            and eph.ftype in prov_list,
            ephem_list,
        )
    )

    return candidate_ephem


def get_preferred_ephemeris_type(
    candidate_ephem_list: List[TruthEphemerisManager], preferred_prov: str
) -> TruthEphemerisManager:
    """Selects preferred ephemeris type, if it exists."""

    eph = filter(lambda eph: eph.ftype == preferred_prov, candidate_ephem_list)

    return next(eph)


def get_ephemeris_for_date(
    ephem_list: List[TruthEphemerisManager],
    year: int,
    month: int,
    day: int,
    prov_list: List[str],
    preferred_prov: str,
) -> TruthEphemerisManager:
    """Fetches the best ephemeris of a certain date. Returns zero is no ephemeris exists."""

    candidate_ephems = locate_ephemerides_for_date(
        ephem_list, year, month, day, prov_list
    )
    if len(candidate_ephems) == 0:
        return 0
    if len(candidate_ephems) == 1:
        return candidate_ephems[0]
    if len(candidate_ephems) > 1:
        preferred_ephem = get_preferred_ephemeris_type(candidate_ephems, preferred_prov)
        if preferred_prov:
            return preferred_prov
        else:
            return candidate_ephems[0]


def fetch_consecutive_ephems(
    ephem_list: List[TruthEphemerisManager],
    year: int,
    month: int,
    day: int,
    prov_list: List[str],
    preferred_prov: str,
) -> tuple[TruthEphemerisManager, TruthEphemerisManager,]:
    """Fetches ephems of the same type for consecutive days, the day of and one day prior."""

    one_year, one_month, one_day = previous_day(year, month, day)

    base_year = int("".join(list(str(year)))[2:])
    one_year = int("".join(list(str(one_year)))[2:])

    base_ephem = get_ephemeris_for_date(
        ephem_list, base_year, month, day, prov_list, preferred_prov
    )
    one_ephem = get_ephemeris_for_date(
        ephem_list, one_year, one_month, one_day, prov_list, preferred_prov
    )

    return base_ephem, one_ephem


def rms(L: List[float]) -> float:
    """Takes a list of values and calculates the RMS value."""

    L2 = np.square(np.asarray(L))
    L2m = np.mean(L2)
    RMS = np.sqrt(L2m)

    return RMS


def day_stats(L: List[tuple]) -> List[tuple[float, float]]:
    """Takes a list containing repeated tuples (here, they are triplets but the function is more general) from a series of observations, breaks it up in n lists
    for each dimension and calculates and returns a list of tuples of RMS and std values of each dimension of this series of observations.

    Args:
        List of arbitrary length of the form [[x1,y1,z1],[x2,y2,z2],...]

    Returns:
        List of the form [(x_rms,x_std),(y_rms,y_std),(z_rms,z_std)]
    """
    rms_list = list(map(rms, zip(*L)))
    std_list = list(map(statistics.stdev, zip(*L)))

    return list(zip(rms_list, std_list))


def prov_mean_diff(
    prov1_eph: TruthEphemerisManager,
    prov2_eph: TruthEphemerisManager,
    epoch_unix: float,
) -> tuple[List[float], List[float], List[float], List[float]]:
    """Calculates differences between two ephemerides over a single day.

    Args:
        prov1_eph: Ephemeris object from the first provider,
        prov2_eph: Ephemeris object from the second provider,
        epoch_unix: Unix time from when the comparison starts

    Returns:
        dpECI: List of position residuals (differences) in ECI coordinates,
        dvECI: List of velocity residuals (differences) in ECI coordinates,
        dpRIC: List of position residuals (differences) in RIC coordinates,
        dvRIC: List of velocity residuals (differences) in RIC coordinates,

        The lists are of the form [[dx1,dy1,dz1],[dx2,dy2,dz2],...] for each time step.

    """
    timestep = 150
    one_day = 24 * 60 * 60 / timestep

    dpECI = []
    dpRIC = []
    dvECI = []
    dvRIC = []

    curr_time = 0

    for _ in range(int(one_day)):
        mean_position = (
            prov1_eph.position_at_unix_time(epoch_unix + curr_time)
            + prov2_eph.position_at_unix_time(epoch_unix + curr_time)
        ) / 2
        mean_velocity = (
            prov1_eph.derived_velocity_at_unix_time(epoch_unix + curr_time)
            + prov2_eph.derived_velocity_at_unix_time(epoch_unix + curr_time)
        ) / 2

        mean_pdiff_ECI = (
            prov1_eph.position_at_unix_time(epoch_unix + curr_time)
            - prov2_eph.position_at_unix_time(epoch_unix + curr_time)
        ) / 2
        mean_vdiff_ECI = (
            prov1_eph.derived_velocity_at_unix_time(epoch_unix + curr_time)
            - prov2_eph.derived_velocity_at_unix_time(epoch_unix + curr_time)
        ) / 2

        RTN = eci_to_rtn_rotation_matrix(
            mean_position,
            mean_velocity,
        )

        mean_pdiff_RIC = np.matmul(RTN, mean_pdiff_ECI)
        mean_vdiff_RIC = np.matmul(RTN, mean_vdiff_ECI)

        dpECI.append(mean_pdiff_ECI)
        dpRIC.append(mean_pdiff_RIC)
        dvECI.append(mean_vdiff_ECI)
        dvRIC.append(mean_vdiff_RIC)
        curr_time += timestep

    return dpECI, dpRIC, dvECI, dvRIC


def compare_different_provider_ephems_over_time(
    ephem_list: List[tephem],
    year: int,
    month: int,
    day: int,
    length: int,
    prov1: str,
    prov2: str,
) -> tuple[
    List[tuple[float, float]],
    List[tuple[float, float]],
    List[tuple[float, float]],
    List[tuple[float, float]],
]:
    """Function that compares ephemeris objects from different providers over a certain period of time.

    Args:
        ephem_list: list of tephem objects
        year, month, day: the epoch of the start of the comparison study
        length: the length in days of the comparison study
        prov1, prov2: file extensions of preferred providers

    Returns:
        ECI_pos_unc: list of tuples with rms and std of day uncertainty in position in the ECI frame,
        ECI_vel_unc: list of tuples with rms and std of day uncertainty in velocity in the ECI frame,
        RIC_pos_unc: list of tuples with rms and std of day uncertainty in position in the RIC frame,
        RIC_vel_unc: list of tuples with rms and std of day uncertainty in velocity in the RIC frame,

        The returned lists are of the form [[(x1_rms,x1_std),(y1_rms,y1_std),(z1_rms,z1_std)],[(x2_rms,x2_std),(y2_rms,y2_std),(z2_rms,z2_std)], ...].
    """
    n_year = year
    n_month = month
    n_day = day

    ECI_pos_unc = []
    ECI_vel_unc = []
    RIC_pos_unc = []
    RIC_vel_unc = []

    for _ in range(length):
        prov1_eph_obj = find_ephem(ephem_list, n_year, n_month, n_day, prov1)
        try:
            print("hts:", prov1_eph_obj.name)
        except:
            prov1_eph_obj = find_ephem(ephem_list, n_year, n_month, n_day, "mcc")
            try:
                print("mcc in place of hts:", prov1_eph_obj.name)
            except:
                print("Missed Day")
                n_year, n_month, n_day = next_day(n_year, n_month, n_day)
                continue

        prov2_eph_obj = find_ephem(ephem_list, n_year, n_month, n_day, prov2)
        try:
            print("sgf:", prov2_eph_obj.name)
        except:
            prov2_eph_obj = find_ephem(ephem_list, n_year, n_month, n_day, "mcc")
            try:
                print("mcc in place of sgf:", prov2_eph_obj.name)
            except:
                print("Missed Day")
                n_year, n_month, n_day = next_day(n_year, n_month, n_day)
                continue
        epoch_unix = calculate_epoch_unix(n_year, n_month, n_day)

        try:
            dpECI, dpRIC, dvECI, dvRIC = prov_mean_diff(
                prov1_eph_obj.ephem, prov2_eph_obj.ephem, epoch_unix
            )
        except:
            n_year, n_month, n_day = next_day(n_year, n_month, n_day)
            continue

        ECI_pos_unc.append(day_stats(dpECI))
        ECI_vel_unc.append(day_stats(dvECI))
        RIC_pos_unc.append(day_stats(dpRIC))
        RIC_vel_unc.append(day_stats(dvRIC))

        n_year, n_month, n_day = next_day(n_year, n_month, n_day)
        print("checked mean")
        print("next month:", n_month)
        print("next day:", n_day)

    return ECI_pos_unc, ECI_vel_unc, RIC_pos_unc, RIC_vel_unc


def single_prov_diff(
    base_eph: TruthEphemerisManager,
    secondary_eph: TruthEphemerisManager,
    epoch_unix: float,
) -> tuple[
    List[tuple[float, float]],
    List[tuple[float, float]],
    List[tuple[float, float]],
    List[tuple[float, float]],
]:
    """Calculates differences between two ephemerides that belong to the same provider.

    Args:
        base_eph: Ephemeride from the day that the comparison will start,
        secondary_eph: Ephemeride from the day before the comparison will start (will be extrapolated to the day of comparison),
        epoch_unix: Time when the comparison will start

    Returns:
        ECI_pos_unc: list of tuples with rms and std of day uncertainty in position in the ECI frame,
        ECI_vel_unc: list of tuples with rms and std of day uncertainty in velocity in the ECI frame,
        RIC_pos_unc: list of tuples with rms and std of day uncertainty in position in the RIC frame,
        RIC_vel_unc: list of tuples with rms and std of day uncertainty in velocity in the RIC frame

        The returned lists are of the form [[(x1_rms,x1_std),(y1_rms,y1_std),(z1_rms,z1_std)],[(x2_rms,x2_std),(y2_rms,y2_std),(z2_rms,z2_std)], ...].
    """
    timestep = 150
    one_day = 24 * 60 * 60 / timestep

    dpECI = []
    dpRIC = []
    dvECI = []
    dvRIC = []

    curr_time = 0

    for _ in range(int(one_day)):
        ECI_pdiff = base_eph.position_at_unix_time(
            epoch_unix + curr_time
        ) - secondary_eph.position_at_unix_time(epoch_unix + curr_time)

        ECI_vdiff = base_eph.derived_velocity_at_unix_time(
            epoch_unix + curr_time
        ) - secondary_eph.derived_velocity_at_unix_time(epoch_unix + curr_time)

        RTN = eci_to_rtn_rotation_matrix(
            base_eph.position_at_unix_time(epoch_unix + curr_time),
            base_eph.derived_velocity_at_unix_time(epoch_unix + curr_time),
        )

        RIC_pdiff = np.matmul(RTN, ECI_pdiff)
        RIC_vdiff = np.matmul(RTN, ECI_vdiff)

        dpECI.append(ECI_pdiff)
        dvECI.append(ECI_vdiff)
        dpRIC.append(RIC_pdiff)
        dvRIC.append(RIC_vdiff)

        curr_time += timestep
    return dpECI, dvECI, dpRIC, dvRIC


def compare_single_prov_ephems_over_time(
    ephem_list: List[tephem],
    year: int,
    month: int,
    day: int,
    length: int,
    prov_list: List[str],
    preferred_prov: str,
):
    """Function that compares ephemeris objects from the same provider over a certain period of time.

    Args:
        ephem_list: list of tephem objects
        year, month, day: the epoch of the start of the comparison study
        length: the length in days of the comparison study
        prov1, prov2: file extensions of preferred providers

    Returns:
        ECI_pos_unc: list of tuples with rms and std of day uncertainty in position in the ECI frame,
        ECI_vel_unc: list of tuples with rms and std of day uncertainty in velocity in the ECI frame,
        RIC_pos_unc: list of tuples with rms and std of day uncertainty in position in the RIC frame,
        RIC_vel_unc: list of tuples with rms and std of day uncertainty in velocity in the RIC frame,

        The returned lists are of the form [[(x1_rms,x1_std),(y1_rms,y1_std),(z1_rms,z1_std)],[(x2_rms,x2_std),(y2_rms,y2_std),(z2_rms,z2_std)], ...].

    """

    n_year = year
    n_month = month
    n_day = day

    ECI_pos_unc = []
    ECI_vel_unc = []
    RIC_pos_unc = []
    RIC_vel_unc = []

    for _ in range(length):
        Eb, E1 = fetch_consecutive_ephems(
            ephem_list, n_year, n_month, n_day, prov_list, preferred_prov
        )
        try:
            Eb.name and E1.name
        except AttributeError:
            print("Missing Day")
            continue  # skip the whole day if one ephemeris is missing

        print("base:", Eb.name)
        print("one:", E1.name)
        epoch_unix = calculate_epoch_unix(n_year, n_month, n_day)

        dpECI, dvECI, dpRIC, dvRIC = single_prov_diff(Eb.ephem, E1.ephem, epoch_unix)

        ECI_pos_unc.append(day_stats(dpECI))
        ECI_vel_unc.append(day_stats(dvECI))
        RIC_pos_unc.append(day_stats(dpRIC))
        RIC_vel_unc.append(day_stats(dvRIC))

        n_year, n_month, n_day = next_day(n_year, n_month, n_day)

    return ECI_pos_unc, ECI_vel_unc, RIC_pos_unc, RIC_vel_unc


def tephem_factory(directory: str, file: str) -> tephem:
    """Instantiates tephem objects from their filename."""
    name = str(file)
    date = file.split("_")[-2]
    year = "".join(list(date)[0:2])
    month = "".join(list(date)[2:4])
    day = "".join(list(date)[4:])
    ftype = file.split(".")[-1]
    if ftype == "dgf":
        eph = 0  # Initialize it with something since TEM breaks with dgf files.
    else:
        eph = TruthEphemerisManager([directory + file])

    return tephem(int(year), int(month), int(day), eph, str(name), str(ftype))


def truth_ephems_from_directory(directory: str) -> List[tephem]:
    """Puts tephem objects in a list."""
    file_list = os.listdir(directory)
    ephemerides = list(map(lambda x: tephem_factory(directory, x), file_list))

    return ephemerides


"""DEPRECATED"""


def exp_fit(uncertainty_list):
    """Returns intercept from exponential fitting of the error between 3 days."""
    days_list = [1, 2, 3]
    _, intercept = np.polyfit(days_list, np.log(uncertainty_list), 1)
    return np.exp(intercept)


def lin_fit(uncertainty_list):
    """Returns intercept from exponential fitting of the error between 3 days."""
    days_list = [1, 2, 3]
    _, intercept = np.polyfit(days_list, uncertainty_list, 1)
    return intercept


def calc_shgo_mode(data):
    """Returns mode of the distribution of single day values. Assumes Gaussian kernel."""
    distribution = scipy.stats.gaussian_kde(data)

    def objective(x):
        return 1 / distribution.pdf(x)[0]

    bnds = [[min(data), max(data)]]
    solution = scipy.optimize.shgo(objective, bounds=bnds, n=100 * len(data))
    return solution.x[0]


def final_uncertainty(days_errors):
    """Calculates the final uncertainty value from the distribution of uncertainties from 30 days of observations."""
    try:
        unc = calc_shgo_mode(days_errors) + 1.96 * (statistics.pstdev(days_errors))
    except:
        unc = "Not enough data"
    return unc


def log_normal_right_confidence_interval(days_errors):
    """Calculates the final uncertianty as a confidence interval (Cox method) from data coming from a log-normal distribution."""
    N = len(days_errors)
    lnX = np.log(days_errors)
    mu = np.mean(lnX)
    sigma = statistics.pstdev(days_errors)

    return np.exp(
        mu + (sigma**2) / 2 + np.sqrt(sigma**2 / N + sigma**4 / (2 * (N - 1)))
    )


def final_uncertainty_gaussian(days_errors):
    """Calculates the final uncertainty value from the distribution of uncertainties from 30 days of observations."""
    try:
        unc = np.mean(days_errors) + 1.96 * (statistics.pstdev(days_errors))
    except:
        unc = "Not enough data"
    return unc
