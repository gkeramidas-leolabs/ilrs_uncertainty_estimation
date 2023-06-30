import pandas as pd
import sys
from pathlib import Path

modules_path = Path("/Users/gkeramidas/Projects/ilrs_uncertainty_estimation")
sys.path.append(str(modules_path))

import ilrs_error_utilities as ieu
import truth_request as tr

targets = ["L1471", "L5429", "L3969"]
# targets = ["L1471", "L5429", "L3972", "L3969", "L2669"]  # single-providers
# targets = ['L2486','L3059','L4884','L5011','L2682'] # multi-providers
# targets = ['L5011']
end_epoch = [2023, 6, 15]
end_year = end_epoch[0]
end_month = end_epoch[1]
end_day = end_epoch[2]
length_of_search = 3
prov_list = ["esa", "cne"]
preferred_prov = "esa"
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
        # sets up directory and downloads relevant data files
        directory = tr.set_up_truth_directory_for_target(leolabs_id, truthdir) + "/"
        tr.dwld_data_for_target(leolabs_id, end_epoch, length_of_search, truthdir)

        # Initializes tephem objects from directory files
        ephemerides = ieu.truth_ephems_from_directory(directory)

        # Main function that does the comparison
        (
            ECI_pos,
            ECI_vel,
            date_list,
        ) = ieu.compare_single_prov_ephems_over_time_economy(
            ephemerides,
            base_year,
            base_month,
            base_day,
            length_of_search,
            prov_list,
            preferred_prov,
        )

        df = ieu.create_dataframe_from_comparison_output_economy(
            ECI_pos, ECI_vel, date_list
        )
        df.to_csv(str(outdir) + "/" + filename, index=False)


if __name__ == "__main__":
    main()
