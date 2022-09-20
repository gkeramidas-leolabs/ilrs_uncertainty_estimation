import ILRS_Uncertainty as unc
import truth_request as tr
import orekit
import random
import numpy as np
from orekit.pyhelpers import  setup_orekit_curdir
from os.path import exists

# Initialize orekit
orekit_vm = orekit.initVM()
setup_orekit_curdir("/Users/gkeramidas/Projects/learning/leolabs-config-data-dynamic/")

targets = ['L1471','L5429','L3972','L3969','L2669'] # single-providers
#targets = ['L2486','L3059','L4884','L5011','L2682'] # multi-providers
#targets = ['L5011']
year_list = [2022,2022,2022,2022]
month_list = [6,7,8,9]
day_list = [30,31,31,18]

for i in range(len(year_list)):
    for leolabs_id in targets:
        filepath = "/Users/gkeramidas/Projects/learning/"+str(leolabs_id)+"-SP-velocity-uncertainties.txt"
        end_epoch = [year_list[i],month_list[i],day_list[i]] # epoch from which we will go back certain number of days
        length_of_search = 30
        prov_list = ["esa","cne"]
        preferred_prov = "esa"

        end_year = end_epoch[0]
        end_month = end_epoch[1]
        end_day = end_epoch[2]

        base_year, base_month, base_day = unc.base_date([end_year, end_month, end_day], length_of_search) # base epoch from where our comparisons begin

        datestring = str(base_month)+"/"+str(base_day)+"/"+str(base_year) 
        
        if exists(filepath):
            pass
        else:
            f=open(filepath,"w")
            f.write(str(leolabs_id) +"\t"+"dVX1"+"\t"+"dVY1"+"\t"+"dVZ1"+"\n")
            f.close()

        print("leo_id:", leolabs_id)
        print(base_year,base_month,base_day)

        # sets up directory and downloads relevant data files
        directory = tr.set_up_truth_directory_for_target(leolabs_id)+"/"
        tr.dwld_data_for_target(leolabs_id, end_epoch, length_of_search)

        # Initializes tephem objects from directory files
        ephemerides = unc.truth_ephems_from_directory(directory)

        # Debugging printing statement
        for ephem in ephemerides:
            print(ephem.name)

        # Main function that does the comparison
        unc_VX1day, unc_VY1day, unc_VZ1day = unc.compare_successive_ephems_velocity(ephemerides, base_year, base_month, base_day, length_of_search, prov_list, preferred_prov)

        
        dVX1d_list = unc_VX1day
        dVY1d_list = unc_VY1day
        dVZ1d_list = unc_VZ1day
        
        dVX1d = np.mean(dVX1d_list)
        dVY1d = np.mean(dVY1d_list)
        dVZ1d = np.mean(dVZ1d_list)

        with open(filepath,"a") as f:
            f.write(datestring + "\t"+ str(dVX1d) + "\t" + str(dVY1d) + "\t" + str(dVZ1d) + "\n")
            