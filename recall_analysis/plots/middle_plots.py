
# import from joint
#%%

combined_example = combined_recalls_intersections[0]

# #%%

# boundies_wo_zeros = combined_example["transition_memories"][
#     combined_example["transition_memories"] != 0
# ]

#%%

arr = np.array(sorted(combined_example["transition_size"]))

#%% Make groups

possible_group_boudaries = np.unique(arr)

#%%
grouped = np.array_split(arr, 10)