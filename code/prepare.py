# LibriCrowd Prepare Data

__author__ = 'jg'

import os, random
import numpy as np
import pandas as pd
import torch
import torchaudio

folder = '../data/'

df_speaker = pd.read_csv(os.path.join(folder, 'LibriCrowd', 'speaker.csv'))
df_speaker.set_index('ID', inplace=True)
df_speaker.info()

exist_speakers = set()
for subset in ('train-other-10h','train-mixed-10h','dev-clean','dev-other','test-clean','test-other'):
    path = os.path.join(folder, 'LibriCrowd', subset)
    for speaker in os.listdir(path):
        if not speaker.isnumeric():
            continue
        exist_speakers.add(int(speaker))

speakers_f = set()
speakers_m = set()
subset = 'train-other-500'
path = os.path.join(folder, 'LibriCrowd', subset)
for speaker in os.listdir(path):
    if speaker not in exist_speakers and speaker.isnumeric():
        if df_speaker.loc[int(speaker),'SEX']=='F':
            speakers_f.add(int(speaker))
        else:
            speakers_m.add(int(speaker))

selected_speakers_f = sorted(random.sample(list(speakers_f), 72))
selected_speakers_m = sorted(random.sample(list(speakers_m), 72))
minutes = 0
for speaker in selected_speakers_f + selected_speakers_m:
    minutes += df_speaker.loc[speaker, 'MINUTES']
print('selected_speakers_f: {:d}, selected_speakers_m: {:d}, hours = {:.2f}'.format(len(selected_speakers_f), len(selected_speakers_m), minutes/60))


# Prepare Ground Truth Data

dataset_name = 'train-clean-100' # train-clean-100, train-other-500, train-other-10h
librispeech_data = torchaudio.datasets.LIBRISPEECH(folder, url=dataset_name, download=False)
data_loader = torch.utils.data.DataLoader(librispeech_data, batch_size=1, shuffle=True, num_workers=1)

print('dataset_name = {:s}, size = {:d}'.format(dataset_name, len(data_loader)))

def groundtruth2std(groundtruth_trans):
    lst_groundtruth_trans = groundtruth_trans.split()
    lst_std_trans  = []
    for word in lst_groundtruth_trans:
        # randomly delete 10%
        if random.random() > 0.1:
            lst_std_trans.append(word)
        # randomly insert (repeat) 10%
        if random.random() > 0.9:
            lst_std_trans.append(word)
    return ' '.join(lst_std_trans)

# groundtruth2std('they were talking confidentially together but when i came down they ceased')

n = len(data_loader)
lst_trans = []
total_length = 0
i = 0
for batch in data_loader:
    waveform = batch[0][0]
    sample_rate = batch[1][0]
    audio_length = batch[0].size()[2] / sample_rate
    total_length += audio_length
    groundtruth_trans = batch[2][0]
    std_trans  = groundtruth2std(groundtruth_trans)
    speaker_id = batch[3][0]
    chapter_id = batch[4][0]
    utterance_id = batch[5][0] 
    utt = '{:d}_{:d}_{:d}'.format(speaker_id, chapter_id, utterance_id)
    # store result
    lst_trans.append([utt, groundtruth_trans, std_trans])
    # display progress
    if i % 10 == 0:
        print('Task ready [{:04d}/{:04d}].'.format(i, n), end = '\r')
    i += 1
    
print('Total length = {:.2f} hours, #utterance = {:d}'.format(total_length/3600, n))

col1 = ['utt','groundtruth_trans','std_trans']
df_trans = pd.DataFrame(lst_trans, columns=col1)
df_trans = df_trans.reset_index(drop=True)
path = os.path.join(folder, 'LibriCrowd', 'LibriCrowd_{:s}_trans.csv'.format(dataset_name.replace('-','_')))
df_trans.to_csv(path, index=False)
df_trans.info()


# Prepare MTurk Data

data_dir = os.path.join(folder, 'LibriCrowd', 'train-clean-100')
lst_data_dir = sorted(os.listdir(data_dir))
dfs = []

for speaker in lst_data_dir:
    if not speaker.isnumeric():
        continue
    chapters = os.listdir(os.path.join(data_dir, speaker))
    lst_audio_files = []
    for chapter in chapters:
        if not chapter.isnumeric():
            continue
        audio_files = os.listdir(os.path.join(data_dir, speaker, chapter))
        lst_audio_files += audio_files
    df = pd.DataFrame(data={'audio_url': [os.path.join(x.split('-')[0], x.split('-')[1], x) for x in lst_audio_files if '.txt' not in x]})
    dfs.append(df)
#         print('speaker = {:s}, num_audio_files = {:03d}'.format(speaker, len(df)))

for i in range(12):
    out_path = os.path.join(data_dir, 'input_{:02d}.csv'.format(i+1))
    df_all = pd.concat(dfs[i*12:i*12+12])
    df_all.to_csv(out_path, index=False)
    
    
# Prepare MTurk CSV index file

def df2bundle(df, r=2):
    d = {}
    for i in range(r):
        d['audio_url_{:d}'.format(i+1)] = []
    lst_audio_url = sorted(df['audio_url'])
    l = len(lst_audio_url)
    lst_audio_url += ['']*(l-l%5)
    for i in range(l//r+1):
        for j in range(r):
            d['audio_url_{:d}'.format(j+1)].append(lst_audio_url[i*r+j])
    return pd.DataFrame(d)

data_dir = os.path.join(folder, 'LibriCrowd', 'train-clean-100')
print('data_dir = {:s}'.format(data_dir))

lst_data_dir = sorted(os.listdir(data_dir))
for speaker in lst_data_dir:
    if not speaker.isnumeric():
        continue
    out_path = os.path.join(data_dir, speaker, 'input.csv')
    chapters = os.listdir(os.path.join(data_dir, speaker))
    lst_audio_files = []
    for chapter in chapters:
        if not chapter.isnumeric():
            continue
        audio_files = os.listdir(os.path.join(data_dir, speaker, chapter))
        lst_audio_files += audio_files
    df = pd.DataFrame(data={'audio_url': [x for x in lst_audio_files if '.txt' not in x]})
    df.to_csv(out_path, index=False)
    print('num_audio_files = {:03d}, out_path = {:s}'.format(len(df), out_path))
    # use bundle
    df2 = df2bundle(df, r=2)
    df5 = df2bundle(df, r=5)
    out_path_2 = os.path.join(data_dir, speaker, 'input2.csv')
    out_path_5 = os.path.join(data_dir, speaker, 'input5.csv')
    df2.to_csv(out_path_2, index=False)
    df5.to_csv(out_path_5, index=False)
    