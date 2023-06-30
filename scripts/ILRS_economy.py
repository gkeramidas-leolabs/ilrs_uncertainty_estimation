import pandas as pd
import sys
from pathlib import Path

modules_path = Path("/Users/gkeramidas/Projects/ilrs_uncertainty_estimation")
sys.path.append(str(modules_path))

import ilrs_error_utilities as ieu
import truth_request as tr


# targets = ["L5011"]
targets = ["L2486", "L3059", "L4884", "L5011", "L2682"]  # multi-providers
end_epoch = [2023, 6, 25]
end_year = end_epoch[0]
end_month = end_epoch[1]
end_day = end_epoch[2]
length_of_search = 365
prov1 = "hts"
prov2 = "sgf"
outdir = Path("/Users/gkeramidas/Projects/ilrs_uncertainty_estimation/results/")
truthdir = Path("/Users/gkeramidas/Projects/ilrs_uncertainty_estimation/")


def main():

    for leolabs_id in targets:
        base_year, base_month, base_day = ieu.base_date(
            [end_year, end_month, end_day], length_of_search
        )  # base epoch from where our comparisons begin

        print("leo_id:", leolabs_id)

        print(base_year, base_month, base_day)

        filename = f"{leolabs_id}-{base_year}-{base_month}-{base_day}-to-{end_year}-{end_month}-{end_day}_economy.csv"

        directory = tr.set_up_truth_directory_for_target(leolabs_id, truthdir) + "/"
        tr.dwld_data_for_target(leolabs_id, end_epoch, length_of_search, truthdir)
        ephemerides = ieu.truth_ephems_from_directory(directory)
        (
            ECI_pos,
            ECI_vel,
            date_list,
        ) = ieu.compare_different_provider_ephems_over_time_economy(
            ephemerides, base_year, base_month, base_day, length_of_search, prov1, prov2
        )
        df = ieu.create_dataframe_from_comparison_output_economy(
            ECI_pos, ECI_vel, date_list
        )
        df.to_csv(str(outdir) + "/" + filename, index=False)


if __name__ == "__main__":
    main()
