import ILRS_Uncertainty as unc
import truth_request as tr
import orekit
from orekit.pyhelpers import setup_orekit_curdir
from os.path import exists

# Initialize orekit
orekit_vm = orekit.initVM()
setup_orekit_curdir("/Users/gkeramidas/Projects/learning/leolabs-config-data-dynamic/")

targets = ["L5011"]
year_list = [2022]
month_list = [8]
day_list = [31]

for i in range(len(year_list)):
    for leolabs_id in targets:
        filepath = (
            "/Users/gkeramidas/Projects/learning/"
            + str(leolabs_id)
            + "-RIC-uncertainties.txt"
        )
        end_epoch = [
            year_list[i],
            month_list[i],
            day_list[i],
        ]  # epoch from which we will go back certain number of days
        length_of_search = 30
        prov1 = "hts"
        prov2 = "sgf"

        end_year = end_epoch[0]
        end_month = end_epoch[1]
        end_day = end_epoch[2]

        base_year, base_month, base_day = unc.base_date(
            [end_year, end_month, end_day], length_of_search
        )  # base epoch from where our comparisons begin

        print("leo_id:", leolabs_id)

        if exists(filepath):
            pass
        else:
            f = open(filepath, "w")
            f.write(
                str(leolabs_id)
                + "\t"
                + "dR1"
                + "\t"
                + "dI1"
                + "\t"
                + "dC1"
                + "\t"
                + "dR2"
                + "\t"
                + "dI2"
                + "\t"
                + "dC2"
                + "\t"
                + "dVR1"
                + "\t"
                + "dVI1"
                + "\t"
                + "dVC1"
                + "\t"
                + "dVR2"
                + "\t"
                + "dVI2"
                + "\t"
                + "dVC2"
                + "\n"
            )
            f.close()

        print(base_year, base_month, base_day)
        datestring = str(base_month) + "/" + str(base_day) + "/" + str(base_year)

        directory = tr.set_up_truth_directory_for_target(leolabs_id) + "/"
        tr.dwld_data_for_target(leolabs_id, end_epoch, length_of_search)
        ephemerides = unc.truth_ephems_from_directory(directory)
        for ephem in ephemerides:
            print(ephem.name)
        dR1, dR2, dI1, dI2, dC1, dC2 = unc.compare_eph_RIC(
            ephemerides, base_year, base_month, base_day, length_of_search, prov1, prov2
        )

        uncR1 = unc.final_uncertainty(dR1)
        uncR2 = unc.final_uncertainty(dR2)
        uncI1 = unc.final_uncertainty(dI1)
        uncI2 = unc.final_uncertainty(dI2)
        uncC1 = unc.final_uncertainty(dC1)
        uncC2 = unc.final_uncertainty(dC2)

        dVR1, dVR2, dVI1, dVI2, dVC1, dVC2 = unc.compare_eph_velocity_RIC(
            ephemerides, base_year, base_month, base_day, length_of_search, prov1, prov2
        )

        uncVR1 = unc.final_uncertainty(dVR1)
        uncVR2 = unc.final_uncertainty(dVR2)
        uncVI1 = unc.final_uncertainty(dVI1)
        uncVI2 = unc.final_uncertainty(dVI2)
        uncVC1 = unc.final_uncertainty(dVC1)
        uncVC2 = unc.final_uncertainty(dVC2)
        print("Finished month for target")
        with open(filepath, "a") as f:
            f.write(
                datestring
                + "\t"
                + str(uncR1)
                + "\t"
                + str(uncI1)
                + "\t"
                + str(uncC1)
                + "\t"
                + str(uncR2)
                + "\t"
                + str(uncI2)
                + "\t"
                + str(uncC2)
                + "\t"
                + str(uncVR1)
                + "\t"
                + str(uncVI1)
                + "\t"
                + str(uncVC1)
                + "\t"
                + str(uncVR2)
                + "\t"
                + str(uncVI2)
                + "\t"
                + str(uncVC2)
                + "\n"
            )
