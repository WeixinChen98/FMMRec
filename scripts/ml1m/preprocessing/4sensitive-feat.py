import os, csv
import pandas as pd

data_dir_path = './'

os.chdir(data_dir_path)
os.getcwd()

inter_file_path = 'ml1m-indexed-v4.inter'
inter_df = pd.read_csv(inter_file_path, sep='\t')
print(f'shape: {inter_df.shape}')

user_file_path = 'users.dat'
user_df = pd.read_csv(user_file_path, names=['user_id', 'gender', 'age', 'occupation', '_3'], header=None, sep='::')
print(f'Shape: {user_df.shape}')

gender_dict = dict(zip(user_df['user_id'], user_df['gender']))
age_dict = dict(zip(user_df['user_id'], user_df['age']))
occupation_dict = dict(zip(user_df['user_id'], user_df['occupation']))

	# *  1:  "Under 18"
	# * 18:  "18-24"
	# * 25:  "25-34"
	# * 35:  "35-44"
	# * 45:  "45-49"
	# * 50:  "50-55"
	# * 56:  "56+"
age_remap = {1:0, 18:1, 25:2, 35:3, 45:4, 50:5, 56:6}


# load user mapping
i_id_mapping = 'u_id_mapping.csv'
mapping_df = pd.read_csv(i_id_mapping, sep='\t')
print(f'shape: {mapping_df.shape}')

# remapping
# map_dict = dict(zip(inter_df['user_id'], inter_df['userID']))
re_map_dict = dict(zip(mapping_df['userID'], mapping_df['user_id']))

# F:0, M:1
gender = []
age = []
occupation = []
for index, row in inter_df.iterrows():
    userID = row['userID']
    g = gender_dict[re_map_dict[userID]]
    assert g == 'F' or g == 'M'
    gender.append(1 if g == 'M' else 0)

    a = age_remap[age_dict[re_map_dict[userID]]]
    age.append(a)
    occupation.append(occupation_dict[re_map_dict[userID]])


inter_df['u_gender'] = gender
inter_df['u_age'] = age
inter_df['u_occupation'] = occupation


new_labeled_file = 'ml1m-indexed-v5.inter'
inter_df.to_csv(os.path.join('./', new_labeled_file), sep='\t', index=False)
print('done!!!')

