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
        filepath = "/Users/gkeramidas/Projects/learning/"+str(leolabs_id)+"-SP-uncertainties.txt"
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
            #f.write(str(leolabs_id) +"\t"+"dX1"+"\t"+"dY1"+"\t"+"dZ1"+"\t"+"dXlin"+"\t"+"dYlin"+"\t"+"dZlin"+"\t"+"dXepx"+"\t"+"dYexp"+"\t"+"dZexp"+"\n")    
            f.write(str(leolabs_id) +"\t"+"dX1"+"\t"+"dY1"+"\t"+"dZ1"+"\n")
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
        unc_X1day, unc_Y1day, unc_Z1day = unc.compare_successive_ephems(ephemerides, base_year, base_month, base_day, length_of_search, prov_list, preferred_prov)

        
        dX1d_list = unc_X1day
        #dX_lin_list = []
        #dX_exp_list = []
        dY1d_list = unc_Y1day
        #dY_lin_list = []
        #dY_exp_list = []
        dZ1d_list = unc_Z1day
        #dZ_lin_list = []
        #dZ_exp_list = []

        dX1d = np.mean(dX1d_list)
        dY1d = np.mean(dY1d_list)
        dZ1d = np.mean(dZ1d_list)

        #for j in range(len(unc_X1day)):
        #    dX_lin_list.append(unc.lin_fit([unc_X1day[j],unc_X2days[j],unc_X3days[j]]))
        #    dY_lin_list.append(unc.lin_fit([unc_Y1day[j],unc_Y2days[j],unc_Y3days[j]]))
        #    dZ_lin_list.append(unc.lin_fit([unc_Z1day[j],unc_Z2days[j],unc_Z3days[j]]))

        #    dX_exp_list.append(unc.exp_fit([unc_X1day[j],unc_X2days[j],unc_X3days[j]]))
        #    dY_exp_list.append(unc.exp_fit([unc_Y1day[j],unc_Y2days[j],unc_Y3days[j]]))
        #    dZ_exp_list.append(unc.exp_fit([unc_Z1day[j],unc_Z2days[j],unc_Z3days[j]]))   

        #dX_lin = unc.final_uncertainty(dX_lin_list)
        #dY_lin = unc.final_uncertainty(dY_lin_list)
        #dZ_lin = unc.final_uncertainty(dZ_lin_list)                       

        #dX_exp = np.mean(dX_exp_list)
        #dY_exp = np.mean(dY_exp_list)
        #dZ_exp = np.mean(dZ_exp_list)                       

        with open(filepath,"a") as f:
            #f.write(datestring + "\t"+ str(dX1d) + "\t" + str(dY1d) + "\t" + str(dZ1d) + "\t" + str(dX_lin) + "\t" + str(dY_lin) + "\t" + str(dZ_lin) + "\t" + str(dX_exp) + "\t" + str(dY_exp) + "\t" + str(dZ_exp) + "\n")
            f.write(datestring + "\t"+ str(dX1d) + "\t" + str(dY1d) + "\t" + str(dZ1d) + "\n")
            