import truth_request as tr
import truth_analysis as ta
import orekit
from orekit.pyhelpers import  setup_orekit_curdir
from matplotlib import pyplot as plt

# Inputs
ILRS_targets = ['L5011', 'L3059', 'L335', 'L2486', 'L4884', 'L1471', 'L5429', 'L3972', 'L3969', 'L2669', 'L3226']
epoch = [2022,5,15]
num_days = 4


# Download the data from S3
#tr.dwld_data_for_all_targets(ILRS_targets,epoch,num_days)

# Perform truth analysis
Ep_Offset, r_err_coll, i_err_coll, c_err_coll = tr.collections_of_truth_state_errors(ILRS_targets,epoch,num_days)

r_std = tr.extract_std_from_error_distributions(r_err_coll)
i_std = tr.extract_std_from_error_distributions(i_err_coll)
c_std = tr.extract_std_from_error_distributions(c_err_coll)

plt.plot(Ep_Offset,r_std,"g",label="radial")
plt.plot(Ep_Offset,c_std,"b",label="cross-track")
plt.plot(Ep_Offset,i_std,"r",label="in-track")
plt.xlabel("Seconds from estimation Epoch")
plt.ylabel("Stdev")
plt.legend()
plt.show()