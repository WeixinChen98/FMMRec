import os, csv
import pandas as pd
import pickle
from tqdm import tqdm


with open('movies_info.pkl', 'rb') as f:
    movies_infos = pickle.load(f)['movies_infos']

print('len(movies_infos): ', len(movies_infos))

data_dir_path = './'

os.chdir(data_dir_path)
os.getcwd()

# load item mapping
i_id_mapping = 'i_id_mapping.csv'
df = pd.read_csv(i_id_mapping, sep='\t')
print(f'shape: {df.shape}')

# remapping
map_dict = dict(zip(df['movie_id'], df['itemID']))

# descriptions: synopsis if available, otherwise plot
i_ids = []
descriptions = []
for movie_id, movie_info in tqdm(movies_infos.items()):
    textual_input = None
    textual_input = movie_info.get('synopsis')
    if textual_input is None:
        plots = movie_info.get('plot')
        if plots is not None:
            textual_input = plots[0]
    else:
        textual_input = textual_input[0]
    
    i_ids.append(movie_id)
    descriptions.append(textual_input)
    # if textual_input is not None:
    #   descriptions.append(textual_input)
    # else:
    #   descriptions.append(" ")

no_description_ids = set(map_dict.keys()) - set(i_ids)
for movie_id in no_description_ids:
    i_ids.append(movie_id)
    descriptions.append(None)

meta_df = pd.DataFrame({'movie_id': i_ids, 'description': descriptions})


meta_df['itemID'] = meta_df['movie_id'].map(map_dict)
meta_df.dropna(subset=['itemID'], inplace=True)
meta_df['itemID'] = meta_df['itemID'].astype('int64')
meta_df['description'] = meta_df['description'].fillna(" ")
meta_df.sort_values(by=['itemID'], inplace=True)

print(f'shape: {meta_df.shape}')


ori_cols = meta_df.columns.tolist()
ret_cols = [ori_cols[-1]] + ori_cols[:-1]
print(f'new column names: {ret_cols}')


ret_df = meta_df[ret_cols]

# dump
ret_df.to_csv(os.path.join('./', 'meta-ml1m.csv'), index=False)
print('done!')


## Reload
indexed_df = pd.read_csv('meta-ml1m.csv')
print(f'shape: {indexed_df.shape}')

i_uni = indexed_df['itemID'].unique()
print(f'# of unique items: {len(i_uni)}')
print('min/max of unique items: {0}/{1}'.format(min(i_uni), max(i_uni)))


from os import listdir
import shutil


# original_path = './poster_small/'
# target_path = './poster_small-indexed/'
# poster_count = 0
# for f in tqdm(listdir(original_path)):
#     if int(f.split('.')[0]) in map_dict:
#       indexed_id = str(map_dict[int(f.split('.')[0])])
#       file_suffix = f.split('.')[-1]
#       shutil.copyfile(original_path + f, target_path + indexed_id + '.' + file_suffix)
#       poster_count += 1

# print('poster_count', poster_count)


from moviepy.editor import AudioFileClip

# def convert_to_wav(filename):
#     clip = AudioFileClip(filename)
#     wav_filename = filename.replace("video/", "audio/").replace(".mp4", ".wav")
#     clip.write_audiofile(wav_filename, fps=5000, bitrate="5k")

original_path = './audios/'
target_path = './audios-indexed/'
audio_count = 0
for f in tqdm(listdir(original_path)):
    if int(f.split('.')[0]) in map_dict:
      indexed_id = str(map_dict[int(f.split('.')[0])])
      file_suffix = f.split('.')[-1]
    #   shutil.copyfile(original_path + f, target_path + indexed_id + '.' + file_suffix)

      clip = AudioFileClip(original_path + f)
      clip.write_audiofile(target_path + indexed_id + '.wav', fps=5000, bitrate="5k")
      audio_count += 1

print('audio_count', audio_count)
