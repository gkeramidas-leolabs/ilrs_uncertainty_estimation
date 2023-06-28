from os.path import exists

import ilrs_error_utilities as ieu
import truth_request as tr


targets = ["L5011"]
year_list = [2023]
month_list = [6]
day_list = [15]

for i in range(len(year_list)):
    for leolabs_id in targets:

        end_epoch = [
            year_list[i],
            month_list[i],
            day_list[i],
        ]  # epoch from which we will go back certain number of days
        length_of_search = 2
        prov1 = "hts"
        prov2 = "sgf"

        end_year = end_epoch[0]
        end_month = end_epoch[1]
        end_day = end_epoch[2]

        base_year, base_month, base_day = ieu.base_date(
            [end_year, end_month, end_day], length_of_search
        )  # base epoch from where our comparisons begin

        print("leo_id:", leolabs_id)

        print(base_year, base_month, base_day)

        directory = tr.set_up_truth_directory_for_target(leolabs_id) + "/"
        tr.dwld_data_for_target(leolabs_id, end_epoch, length_of_search)
        ephemerides = ieu.truth_ephems_from_directory(directory)
        for ephem in ephemerides:
            print(ephem.name)
        (
            ECI_pos,
            ECI_vel,
            RIC_pos,
            RIC_unc,
        ) = ieu.compare_different_provider_ephems_over_time(
            ephemerides, base_year, base_month, base_day, length_of_search, prov1, prov2
        )
