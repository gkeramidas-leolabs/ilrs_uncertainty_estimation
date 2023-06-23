import os
from datetime import datetime
import statistics
from collections import namedtuple
from typing import List, Dict, Any, Union

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
    """Takes one day and returns the next one. No leap years."""

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
    """Takes an epoch and returns the previous day. No leap years."""

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

        return get_preferred_ephemeris_type(candidate_ephems, preferred_prov)


def fetch_consecutive_ephems(
    ephem_list: List[TruthEphemerisManager],
    year: int,
    month: int,
    day: int,
    prov_list: List[str],
    preferred_prov: str,
) -> tuple[
    TruthEphemerisManager,
    TruthEphemerisManager,
    TruthEphemerisManager,
    TruthEphemerisManager,
]:
    """Fetches ephems of the same type for consecutive days."""

    one_year, one_month, one_day = previous_day(year, month, day)
    two_year, two_month, two_day = previous_day(one_year, one_month, one_day)
    three_year, three_month, three_day = previous_day(two_year, two_month, two_day)

    year = int("".join(list(str(year)))[2:])
    one_year = int("".join(list(str(one_year)))[2:])
    two_year = int("".join(list(str(two_year)))[2:])
    three_year = int("".join(list(str(three_year)))[2:])

    base_ephem = get_ephemeris_for_date(
        ephem_list, year, month, day, prov_list, preferred_prov
    )
    one_ephem = get_ephemeris_for_date(
        ephem_list, one_year, one_month, one_day, prov_list, preferred_prov
    )
    two_ephem = get_ephemeris_for_date(
        ephem_list, two_year, two_month, two_day, prov_list, preferred_prov
    )
    three_ephem = get_ephemeris_for_date(
        ephem_list, three_year, three_month, three_day, prov_list, preferred_prov
    )

    return base_ephem, one_ephem, two_ephem, three_ephem


def rms(L: List[float]) -> float:
    """Takes a list of values and calculates the RMS value."""
    L2 = np.square(np.asarray(L))
    L2m = np.mean(L2)
    RMS = np.sqrt(L2m)

    return RMS


def day_stats(L: List[tuple]) -> List[tuple[float, float]]:
    """Takes a list containing repeated tuples (here, they are triplets but the function is more general) from a series of observations, breaks it up in n lists
    for each dimension and calculates and returns a list of tuples of RMS and std values of each dimension of this series of observations.
    The input list is of the form [[x1,y1,z1],[x2,y2,z2],...] and the output list is of the form [(x_rms,x_std),(y_rms,y_std),(z_rms,z_std)].
    """
    rms_list = list(map(rms, zip(*L)))
    std_list = list(map(statistics.stdev, zip(*L)))

    return list(zip(rms_list, std_list))


def compare_eph(
    ephem_list: List[tephem],
    year: int,
    month: int,
    day: int,
    length: int,
    prov1: str,
    prov2: str,
):
    """Function that compares ephemeris objects from different providers over a certain period of time.
    It return 4 lists (ECI/RIC position/velocity) which contain tuples of (RMS,std) of each dimension for each of the days over which the comparison takes place.
    The returned lists look like [[(x1_rms,x1_std),(y1_rms,y1_std),(z1_rms,z1_std)],[(x2_rms,x2_std),(y2_rms,y2_std),(z2_rms,z2_std)], ...].
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


def single_prov_diff(base_eph, secondary_eph, epoch_unix):
    """Calculates differences between two ephemerides."""
    timestep = 150
    one_day = 24 * 60 * 60 / timestep

    dX = []
    dY = []
    dZ = []
    dR = []
    dI = []
    dC = []

    curr_time = 0

    for i in range(int(one_day)):
        ECI_diff = base_eph.position_at_unix_time(
            epoch_unix + curr_time
        ) - secondary_eph.position_at_unix_time(epoch_unix + curr_time)

        RTN = eci_to_rtn_rotation_matrix(
            base_eph.position_at_unix_time(epoch_unix + curr_time),
            base_eph.derived_velocity_at_unix_time(epoch_unix + curr_time),
        )

        RIC_diff = np.matmul(RTN, ECI_diff)

        dX.append(ECI_diff[0])
        dY.append(ECI_diff[1])
        dZ.append(ECI_diff[2])

        dR.append(RIC_diff[0])
        dI.append(RIC_diff[1])
        dC.append(RIC_diff[2])

        curr_time += timestep
    return dX, dY, dZ, dR, dI, dC


def single_prov_vel_diff(base_eph, secondary_eph, epoch_unix):
    """Calculates velocity differences between two ephemerides."""
    timestep = 150
    one_day = 24 * 60 * 60 / timestep

    dVX = []
    dVY = []
    dVZ = []
    dVR = []
    dVI = []
    dVC = []

    curr_time = 0

    for i in range(int(one_day)):
        ECI_vel_diff = base_eph.derived_velocity_at_unix_time(
            epoch_unix + curr_time
        ) - secondary_eph.derived_velocity_at_unix_time(epoch_unix + curr_time)

        RTN = eci_to_rtn_rotation_matrix(
            base_eph.position_at_unix_time(epoch_unix + curr_time),
            base_eph.derived_velocity_at_unix_time(epoch_unix + curr_time),
        )

        RIC_vel_diff = np.matmul(RTN, ECI_vel_diff)

        dVX.append(ECI_vel_diff[0])
        dVY.append(ECI_vel_diff[1])
        dVZ.append(ECI_vel_diff[2])

        dVR.append(RIC_vel_diff[0])
        dVI.append(RIC_vel_diff[1])
        dVC.append(RIC_vel_diff[2])

        curr_time += timestep
    return dVX, dVY, dVZ, dVR, dVI, dVC


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


def prov_mean_diff(
    prov1_eph: TruthEphemerisManager,
    prov2_eph: TruthEphemerisManager,
    epoch_unix: float,
) -> tuple[List[float], List[float], List[float], List[float]]:
    """Calculates differences between two ephemerides."""
    timestep = 150
    one_day = 24 * 60 * 60 / timestep

    dpECI = []
    dpRIC = []
    dvECI = []
    dvRIC = []

    curr_time = 0

    for i in range(int(one_day)):
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


def compare_single_prov_ephems(
    ephem_list, year, month, day, length, prov_list, preferred_prov
):
    """Compares ephems from 3 consecutive days over a single day."""

    n_year = year
    n_month = month
    n_day = day

    unc_X1day = []
    unc_X2days = []
    unc_X3days = []
    unc_Y1day = []
    unc_Y2days = []
    unc_Y3days = []
    unc_Z1day = []
    unc_Z2days = []
    unc_Z3days = []

    for i in range(length):
        Eb, E1, E2, E3 = fetch_consecutive_ephems(
            ephem_list, n_year, n_month, n_day, prov_list, preferred_prov
        )
        try:
            Eb.name and E1.name and E2.name and E3.name
        except AttributeError:
            print("Missing Day")
            continue  # skip the whole day if one ephemeris is missing

        print("base:", Eb.name)
        print("one:", E1.name)
        print("two:", E2.name)
        print("three:", E3.name)
        epoch_unix = calculate_epoch_unix(n_year, n_month, n_day)

        dX1, dY1, dZ1, dR1, dI1, dC1 = single_prov_diff(Eb.ephem, E1.ephem, epoch_unix)
        dX2, dY2, dZ2, dR2, dI2, dC2 = single_prov_diff(Eb.ephem, E2.ephem, epoch_unix)
        dX3, dY3, dZ3, dR3, dI3, dC3 = single_prov_diff(Eb.ephem, E3.ephem, epoch_unix)

        # ts = [x for x in range(len(dX1))]
        # fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,5))

        # ax1.plot(ts,dX1,'r',label="one day")
        # ax1.plot(ts,dX2,'g',label="two days")
        # ax1.plot(ts,dX3,'b',label="three days")
        # ax2.plot(ts,dY1,'r',label="one day")
        # ax2.plot(ts,dY2,'g',label="two days")
        # ax2.plot(ts,dY3,'b',label="three days")
        # ax3.plot(ts,dZ1,'r',label="one day")
        # ax3.plot(ts,dZ2,'g',label="two days")
        # ax3.plot(ts,dZ3,'b',label="three days")

        # ax1.set_xlabel("time steps")
        # ax2.set_xlabel("time steps")
        # ax3.set_xlabel("time steps")
        # ax1.set_ylabel("dX (m)")
        # ax2.set_ylabel("dY (m)")
        # ax3.set_ylabel("dZ (m)")
        # plt.legend()
        # plt.show()

        unc_X1day.append(abs(max_conf_day(dX1, 95)))
        unc_Y1day.append(abs(max_conf_day(dY1, 95)))
        unc_Z1day.append(abs(max_conf_day(dZ1, 95)))
        unc_X2days.append(abs(max_conf_day(dX2, 95)))
        unc_Y2days.append(abs(max_conf_day(dY2, 95)))
        unc_Z2days.append(abs(max_conf_day(dZ2, 95)))
        unc_X3days.append(abs(max_conf_day(dX3, 95)))
        unc_Y3days.append(abs(max_conf_day(dY3, 95)))
        unc_Z3days.append(abs(max_conf_day(dZ3, 95)))

        n_year, n_month, n_day = next_day(n_year, n_month, n_day)
    return (
        unc_X1day,
        unc_X2days,
        unc_X3days,
        unc_Y1day,
        unc_Y2days,
        unc_Y3days,
        unc_Z1day,
        unc_Z2days,
        unc_Z3days,
    )


def compare_successive_ephems(
    ephem_list, year, month, day, length, prov_list, preferred_prov
):
    """Compares ephems from 3 consecutive days over a single day."""
    n_year = year
    n_month = month
    n_day = day

    unc_X1day = []
    unc_Y1day = []
    unc_Z1day = []

    for i in range(length):
        Eb, E1, _, _ = fetch_consecutive_ephems(
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

        dX1, dY1, dZ1, dR1, dI1, dC1 = single_prov_diff(Eb.ephem, E1.ephem, epoch_unix)

        unc_X1day.append(abs(max_conf_day(dX1, 95)))
        unc_Y1day.append(abs(max_conf_day(dY1, 95)))
        unc_Z1day.append(abs(max_conf_day(dZ1, 95)))

        n_year, n_month, n_day = next_day(n_year, n_month, n_day)
    return unc_X1day, unc_Y1day, unc_Z1day


def compare_successive_ephems_velocity(
    ephem_list, year, month, day, length, prov_list, preferred_prov
):
    """Compares ephems from 3 consecutive days over a single day."""
    n_year = year
    n_month = month
    n_day = day

    unc_VX1day = []
    unc_VY1day = []
    unc_VZ1day = []

    for i in range(length):
        Eb, E1, _, _ = fetch_consecutive_ephems(
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

        dVX1, dVY1, dVZ1, dVR1, dVI1, dVC1 = single_prov_vel_diff(
            Eb.ephem, E1.ephem, epoch_unix
        )

        unc_VX1day.append(abs(max_conf_day(dVX1, 95)))
        unc_VY1day.append(abs(max_conf_day(dVY1, 95)))
        unc_VZ1day.append(abs(max_conf_day(dVZ1, 95)))

        n_year, n_month, n_day = next_day(n_year, n_month, n_day)
    return unc_VX1day, unc_VY1day, unc_VZ1day


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
