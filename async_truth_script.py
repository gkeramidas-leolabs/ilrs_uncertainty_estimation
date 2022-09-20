import truth_request as tr
import truth_analysis as ta
import orekit
from orekit.pyhelpers import  setup_orekit_curdir
from matplotlib import pyplot as plt
import asyncio
from itertools import zip_longest

# Inputs
ILRS_targets = ['L5011', 'L3059', 'L2486', 'L4884', 'L1471', 'L5429', 'L3972', 'L3969', 'L2669', 'L2682']
epoch = [2022,5,15]
num_days = 2


async def exp_backoff_async_api_call(state_array,start_point,end_point,max_retries=5):
    """Function that performs API calls asynchronously but with exponential backoff strategy to deal with API errors."""
    retries = 0
    while retries < max_retries:
        try:
            await asyncio.sleep(2**retries)
            async_res = await asyncio.gather(*[tr.async_state_requests(state_array[i][0],state_array[i][1],state_array[i][2]) for i in range(start_point,end_point)])
            print("Success!")
            break
        except:
            retries +=1
            print("Failed, retry:", retries)
    return async_res

async def batch_requests(state_array,max_requests=100):
    """Function that makes asynchronous requests in batches."""
    num_req = state_array.shape[0]
    
    num_of_chunks = int(num_req/max_requests)
    
    TOS = []
    for i in range(num_of_chunks+1):
        if i < (num_of_chunks):
            batch_tos = await exp_backoff_async_api_call(state_array, i*max_requests, (i+1)*max_requests,5)
            TOS.extend(batch_tos)
        else:
            batch_tos = await exp_backoff_async_api_call(state_array, i*max_requests, num_req,5)
            TOS.extend(batch_tos)
    return TOS

async def main():

    # Download files
    tr.dwld_data_for_all_targets(ILRS_targets[:],epoch,num_days)

    state_array = tr.collect_all_states(ILRS_targets, epoch, num_days) # collecting all states for the specified date range

    r_err_list = []
    i_err_list = []
    c_err_list = []
    Ep_Offset_list = []

    TOS = await batch_requests(state_array,100) # Doing all API calls and initializing all truth objects

    for i in range(len(TOS)):
        epoch_Offset, r_err, i_err, c_err = tr.truth_analysis_errors(TOS[i])

        if (r_err is not None):
                Ep_Offset_list.append(epoch_Offset)
                r_err_list.append(r_err)
                i_err_list.append(i_err)
                c_err_list.append(c_err)

        else: 
            pass   
        # convert the lists of errors with len = number_of_time_steps to lists of lists of the same length 
        # but each entry is a list with all the errors of that time step.
        r_err_collection = list(zip_longest(*r_err_list)) 
        i_err_collection = list(zip_longest(*i_err_list))
        c_err_collection = list(zip_longest(*c_err_list))

    r_std = tr.extract_std_from_error_distributions(r_err_collection)
    i_std = tr.extract_std_from_error_distributions(i_err_collection)
    c_std = tr.extract_std_from_error_distributions(c_err_collection)

    plt.plot(Ep_Offset_list[0],r_std,"g",label="radial")
    plt.plot(Ep_Offset_list[0],c_std,"b",label="cross-track")
    plt.plot(Ep_Offset_list[0],i_std,"r",label="in-track")
    plt.xlabel("Seconds from estimation Epoch")
    plt.ylabel("Stdev")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    asyncio.run(main())