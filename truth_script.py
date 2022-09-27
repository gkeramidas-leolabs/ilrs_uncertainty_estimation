import truth_request as tr
import truth_analysis as ta
import orekit
from orekit.pyhelpers import  setup_orekit_curdir
from matplotlib import pyplot as plt

# Inputs
ILRS_targets = ['L5011', 'L3059', 'L2486', 'L4884', 'L1471', 'L5429', 'L3972', 'L3969', 'L2669', 'L2682']
#ILRS_targets = ['L5011', 'L3059', 'L2486', 'L4884', 'L1471', 'L2669', 'L2682']
epoch = [2022,5,15]
num_days = 2

# Download files
tr.dwld_data_for_all_targets(ILRS_targets[:],epoch,num_days)

# Perform truth analysis
Ep_Offset, r_err_coll, i_err_coll, c_err_coll, Vr_err_coll, Vi_err_coll, Vc_err_coll  = tr.collections_of_truth_state_errors(ILRS_targets[:],epoch,num_days)

r_std = tr.extract_std_from_error_distributions(r_err_coll)
i_std = tr.extract_std_from_error_distributions(i_err_coll)
c_std = tr.extract_std_from_error_distributions(c_err_coll)
Vr_std = tr.extract_std_from_error_distributions(Vr_err_coll)
Vi_std = tr.extract_std_from_error_distributions(Vi_err_coll)
Vc_std = tr.extract_std_from_error_distributions(Vc_err_coll)

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,5))
ax1.plot(Ep_Offset_list[0],r_std,"g",label="radial")
ax1.plot(Ep_Offset_list[0],c_std,"b",label="cross-track")
ax1.plot(Ep_Offset_list[0],i_std,"r",label="in-track")
ax2.plot(Ep_Offset_list[0],Vr_std,"g",label="radial")
ax2.plot(Ep_Offset_list[0],Vc_std,"b",label="cross-track")
ax2.plot(Ep_Offset_list[0],Vi_std,"r",label="in-track")

ax1.set_xlabel("Seconds from estimation Epoch")
ax1.set_ylabel("Stdev")
ax2.set_xlabel("Seconds from estimation Epoch")
ax2.set_ylabel("Stdev")

ax1.legend()
plt.show()