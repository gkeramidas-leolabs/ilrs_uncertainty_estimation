import ILRS_Uncertainty as unc
import truth_request as tr
import orekit
from orekit.pyhelpers import  setup_orekit_curdir
from os.path import exists

# Initialize orekit
orekit_vm = orekit.initVM()
setup_orekit_curdir("/Users/gkeramidas/Projects/learning/leolabs-config-data-dynamic/")

targets = ['L2486','L3059','L4884','L5011','L2682']
year_list = [2022,2022,2022,2022]
month_list = [6,7,8,9]
day_list = [30,31,31,18]

for i in range(len(year_list)):
    for leolabs_id in targets:
        filepath = "/Users/gkeramidas/Projects/learning/"+str(leolabs_id)+"-velocity-uncertainties.txt"
        end_epoch = [year_list[i],month_list[i],day_list[i]] # epoch from which we will go back certain number of days
        length_of_search = 30
        prov1 = "hts"
        prov2 = "sgf"

        end_year = end_epoch[0]
        end_month = end_epoch[1]
        end_day = end_epoch[2]

        base_year, base_month, base_day = unc.base_date([end_year, end_month, end_day], length_of_search) # base epoch from where our comparisons begin

        print("leo_id:", leolabs_id)


        if exists(filepath):
            pass
        else:
            f=open(filepath,"w")
            f.write(str(leolabs_id) +"\t"+"dVX1"+"\t"+"dVY1"+"\t"+"dVZ1"+"\t"+"dVX2"+"\t"+"dVY2"+"\t"+"dVZ2"+"\n")    
            f.close()

        print(base_year,base_month,base_day)
        datestring = str(base_month)+"/"+str(base_day)+"/"+str(base_year) 

        directory = tr.set_up_truth_directory_for_target(leolabs_id)+"/"
        tr.dwld_data_for_target(leolabs_id, end_epoch, length_of_search)
        ephemerides = unc.truth_ephems_from_directory(directory)
        for ephem in ephemerides:
            print(ephem.name)
        dVX1, dVX2, dVY1, dVY2, dVZ1, dVZ2 = unc.compare_eph_velocity(ephemerides, base_year, base_month, base_day, length_of_search, prov1, prov2)

        uncVX1 = unc.final_uncertainty(dVX1)
        uncVX2 = unc.final_uncertainty(dVX2)
        uncVY1 = unc.final_uncertainty(dVY1)
        uncVY2 = unc.final_uncertainty(dVY2)
        uncVZ1 = unc.final_uncertainty(dVZ1)
        uncVZ2 = unc.final_uncertainty(dVZ2)
        print("Finished month for target")
        with open(filepath,"a") as f:
            f.write(datestring +"\t"+str(uncVX1)+"\t"+str(uncVY1)+"\t"+str(uncVZ1)+"\t"+str(uncVX2)+"\t"+str(uncVY2)+"\t"+str(uncVZ2)+"\n")
        