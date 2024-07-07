import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tools.json_pickle_stuff import read_pickle
from tools.paths_dirs_stuff import path_contents_pattern


log_path = "/home/mehdi/Data/AutoPET24/postprocessing/data_stats"
log_files = path_contents_pattern(log_path, ".pkl")

tumor_size = []
min_ct = []
max_ct = []
median_ct = []
mean_ct = []
min_pt = []
max_pt = []
median_pt = []
mean_pt = []
for ix, log in enumerate(log_files):
    log_abs_path = os.path.join(log_path, log)
    log_dict = read_pickle(log_abs_path)
    for keys, vals in log_dict.items():
        n_components = vals['n_components']
        if n_components > 0:
            tumor_size.append(vals['region_size'])
            min_ct.append(vals['min_ct_vals'])
            max_ct.append(vals['max_ct_vals'])
            median_ct.append(vals['med_ct_vals'])
            min_pt.append(vals['min_pt_vals'])
            max_pt.append(vals['max_pt_vals'])
            median_pt.append(vals['med_pt_vals'])
            mean_ct.append(vals['mean_ct_vals'])
            mean_pt.append(vals['mean_pt_vals'])

def flatten_nested_list(mylist):
    return [x for xx in mylist for x in xx]

tumor_size_arr = np.array(flatten_nested_list(tumor_size))
min_ct_arr = np.array(flatten_nested_list(min_ct))
max_ct_arr = np.array(flatten_nested_list(max_ct))
median_ct_arr = np.array(flatten_nested_list(median_ct))
min_pt_arr = np.array(flatten_nested_list(min_pt))
max_pt_arr = np.array(flatten_nested_list(max_pt))
median_pt_arr = np.array(flatten_nested_list(median_pt))
mean_ct_arr = np.array(flatten_nested_list(mean_ct))
mean_pt_arr = np.array(flatten_nested_list(mean_pt))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
def plot_beta_hist(ax, data, n_bins, alpha, color, edgecolor):
    ax.hist(
        data,
        bins=n_bins,
        alpha=alpha,
        color=color,
        edgecolor=edgecolor,
        density=False)
num_of_bins = 20
size_copy = deepcopy(tumor_size_arr)
ind1 = np.where(tumor_size_arr < 10)
ind2 = np.where((tumor_size_arr>=10) & (tumor_size_arr<20))
ind3 = np.where((tumor_size_arr>=20) & (tumor_size_arr<50))
ind4 = np.where((tumor_size_arr>=50) & (tumor_size_arr<100))
# ind5 = np.where((tumor_size_arr>=100) & (tumor_size_arr<500))
# ind6 = np.where((tumor_size_arr>=500) & (tumor_size_arr<1000))
# ind7 = np.where((tumor_size_arr>=1000) & (tumor_size_arr<6000))
size_grou1 = size_copy[ind1]
size_grou2 = size_copy[ind2]
size_grou3 = size_copy[ind3]
size_grou4 = size_copy[ind4]
# size_grou5 = size_copy[ind5]
# size_grou6 = size_copy[ind6]
# size_grou7 = size_copy[ind7]

plt.style.use('bmh')
fig, ax = plt.subplots()
plot_beta_hist(ax, size_grou1, num_of_bins//2, 0.9, 'royalblue', 'black' )
plot_beta_hist(ax, size_grou2, num_of_bins//2, 0.9, 'purple', 'black' )
plot_beta_hist(ax, size_grou3, num_of_bins//2, 0.9, 'green', 'black' )
plot_beta_hist(ax, size_grou4, num_of_bins//2, 0.9, 'red', 'black' )
# plot_beta_hist(ax, size_grou5, num_of_bins//2, 0.9, 'purple', 'black' )
# plot_beta_hist(ax, size_grou6, num_of_bins//2, 0.9, 'purple', 'black' )
# plot_beta_hist(ax, size_grou7, num_of_bins//2, 0.9, 'purple', 'black' )
ax.set_title("Distribution of tumor size")
ax.set_xlabel("number of occupied voxels")
ax.set_ylabel("Frequenct")
ax.legend(['<10', '10<x<20', '20<x<50', '50<x<100', '100<x<500', '500<x<1000', 'x>1000'])
# plt.savefig('/home/mehdi/Data/AutoPET24/postprocessing/data_stats/histogram.png')
plt.show()


mean_min_pt_g1 = np.mean(min_pt_arr[ind1])
min_min_ptg1 = np.min(min_pt_arr[ind1])

mean_mean_pt_g1 = np.mean(mean_pt_arr[ind1])
min_mean_pt_g1 = np.min(mean_pt_arr[ind1])
perntile1_mean_pt_g1 = np.percentile(mean_pt_arr[ind1],1)
print("percentile 1 for PET average intensity is {:.4f}".format(perntile1_mean_pt_g1))