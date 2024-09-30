import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from imdb import IMDb

data_dir_path = './'

os.chdir(data_dir_path)
os.getcwd()


i_id, desc_str = 'itemID', 'description'

file_path = './'
file_name = 'meta-ml1m.csv'

meta_file = os.path.join(file_path, file_name)

df = pd.read_csv(meta_file)
df.sort_values(by=[i_id], inplace=True)

print('data loaded!')
print(f'shape: {df.shape}')


# sentences: description
df['description'] = df['description'].fillna(" ")

course_list = df[i_id].tolist()
sentences = df[desc_str].tolist()

assert course_list[-1] == len(course_list) - 1
# print(sentences[:10])




from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentence_embeddings = model.encode(sentences)


print('text encoded!')

assert sentence_embeddings.shape[0] == df.shape[0]
np.save(os.path.join(file_path, 'text_feat.npy'), sentence_embeddings)
print('done!')

print(sentence_embeddings[:10])


load_txt_feat = np.load('text_feat.npy', allow_pickle=True)
print(load_txt_feat.shape)
print(load_txt_feat[:10])





# # Image encoder (V0)，following LATTICE, averaging over for missed items

# from os import listdir
# image_dir_path = './poster_small-indexed/'
# item_list = course_list


# image_files = {}
# for f in tqdm(listdir(image_dir_path)):
#     image_files[int(f.split('.')[0])] = image_dir_path + f
    

# import os, pickle, json
# import numpy as np
# import torchvision as tv
# from PIL import Image
# import torch
# import torchvision.transforms as transforms
# import torchvision.models as models

# transform = transforms.Compose([
#         tv.transforms.Resize((224, 224)),
#         tv.transforms.ToTensor(),
#         tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ]
# )


# feats = {}
# avg = []



# image_num = len(image_files)
# image_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) 
# image_tensor = torch.zeros(1, 3, 224, 224)
# for i_id in tqdm(image_files):
#     img = torch.from_numpy(np.array(transform(Image.open(image_files[i_id]).convert('RGB'))))
#     image_tensor[0] = img
#     feat = image_model(image_tensor)[0]
#     feats[i_id] = feat.detach().numpy()
#     avg.append(feat.detach().numpy())

# avg = np.array(avg).mean(0).tolist()

# no_img_ids = set(item_list) - set(image_files.keys())
# print('# of items not in processed image features:', len(no_img_ids))
# assert (len(feats) + len(no_img_ids)) == df.shape[0]

# ret = []
# for i_id in tqdm(item_list):
#     if i_id in feats:
#         ret.append(feats[i_id])
#     else:
#         ret.append(avg)
#         if i_id not in no_img_ids:
#             print('Error!!!')

# assert len(ret) == df.shape[0]

# np.save('image_feat.npy', np.array(ret))
# # np.savetxt("missed_img_itemIDs.txt", no_img_ids, delimiter =",", fmt ='%d')
# with open('missed_img_itemIDs.txt', 'w') as f:
#     for line in no_img_ids:
#         f.write("%d\n" % line)

# print('done!')





# from towhee import AutoPipes
# import torch

# audio_dir_path = './audios-indexed/'

# audio_files = {}
# for f in tqdm(listdir(audio_dir_path)):
#     audio_files[int(f.split('.')[0])] = audio_dir_path + f

# audio_num = len(audio_files)


# feats = {}
# avg = []

# embedding_pipeline = AutoPipes.pipeline('towhee/audio-embedding-vggish')


# for i_id in tqdm(audio_files):
#     feat = embedding_pipeline(audio_files[i_id]).get()
#     feat = torch.mean(torch.tensor(feat).squeeze(), dim = 0)
#     feats[i_id] = feat.detach().numpy()
#     avg.append(feat.detach().numpy())


# avg = np.array(avg).mean(0).tolist()


# no_audio_ids = set(item_list) - set(audio_files.keys())
# print('# of items not in processed audio features:', len(no_audio_ids))
# assert (len(feats) + len(no_audio_ids)) == df.shape[0]

# ret = []
# for i_id in tqdm(item_list):
#     if i_id in feats:
#         ret.append(feats[i_id])
#     else:
#         ret.append(avg)
#         if i_id not in no_audio_ids:
#             print('Error!!!')

# assert len(ret) == df.shape[0]

# np.save('audio_feat.npy', np.array(ret))
# with open('missed_audio_itemIDs.txt', 'w') as f:
#     for line in no_audio_ids:
#         f.write("%d\n" % line)

# print('done!')