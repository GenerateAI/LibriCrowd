# LibriCrowd Relabeling Script

__author__ = 'jg'

import os, re, string
import numpy as np
import pandas as pd
from utils import *

dataset_name = 'train_other_60h'
folder = '../data/'
process = 'process' # process, process2, process5
data_dir = os.path.join(folder, 'LibriCrowd', 'batch_result', process)

batch_nums = sorted([int(file.split('.csv')[0]) for file in os.listdir(data_dir) if '.csv' in file and '_out' not in file])
print('batch_nums =', batch_nums)

num2words = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', \
             6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten', \
            11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', \
            15: 'Fifteen', 16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', \
            19: 'Nineteen', 20: 'Twenty', 30: 'Thirty', 40: 'Forty', \
            50: 'Fifty', 60: 'Sixty', 70: 'Seventy', 80: 'Eighty', \
            90: 'Ninety', 0: 'Zero'}

strnum2word = dict()
for k in num2words:
    strnum2word[str(k)] = num2words[k]


# Compare predict and submit

df_predict = pd.read_csv(os.path.join(folder, 'LibriCrowd', 'LibriCrowd_{:s}_trans.csv'.format(dataset_name)))
df_submits = []
df_submit_aggs = []
for batch_num in batch_nums:
    df_submit = pd.read_csv(os.path.join(folder, 'LibriCrowd', 'batch_result/{:s}/{:d}.csv'.format(process, batch_num)))
    ori_cols = df_submit.columns
    df_submit_agg = df_submit
    df_submit = df_submit[df_submit['Input.audio_url'].apply(lambda x: '.txt' not in x)]
    df_submits.append(df_submit)
    df_submit_agg['batch_num'] = batch_num
    df_submit_aggs.append(df_submit_agg)
df_submit_all = pd.concat(df_submit_aggs).reset_index(drop=True)

df_submit_all = df_submit_all[df_submit_all['AssignmentStatus'] == 'Submitted']
df_submit_all['utt'] = df_submit_all['Input.audio_url'].apply(lambda x: audio2utt(x))
df_check = df_submit_all.merge(df_predict[['utt','asr_trans']], on = ['utt'], how = 'left')
df_check = df_check.rename(columns={'Answer.transcription':'std_trans'}).reset_index(drop=True)
df_check['std_trans'] = df_check['std_trans'].fillna('ImputedNA')
df_check['num_q_mark'] = df_check['std_trans'].apply(lambda x: x.count('?'))
df_check['asr_trans'] = df_check['asr_trans'].apply(lambda x: normalize_trans(x))
df_check['std_trans'] = df_check['std_trans'].apply(lambda x: normalize_trans(x))
if len(df_check) == 0:
    print('No pending task in batch_nums:', batch_nums)

df_check['asr_trans_stc'] = df_check['asr_trans'].apply(lambda x: len(x.split()))
df_check['std_trans_stc'] = df_check['std_trans'].apply(lambda x: len(x.split()))
df_check['std_trans_wec'] = df_check.apply(lambda x: compute_wer(x.asr_trans, x.std_trans), axis=1)

df6 = df_check.groupby('utt').agg({'std_trans_wec': ['mean', 'min', 'max']})
df7 = df_check.groupby('utt').agg({'asr_trans_stc': ['min']})
wer_mean = df6[('std_trans_wec', 'mean')].sum() / df7[('asr_trans_stc', 'min')].sum()
wer_max = df6[('std_trans_wec', 'max')].sum() / df7[('asr_trans_stc', 'min')].sum()
wer_oracle = df6[('std_trans_wec', 'min')].sum() / df7[('asr_trans_stc', 'min')].sum()
print('wer_mean = {:.4f}, wer_oracle = {:.4f}, wer_max = {:.4f}'.format(wer_mean, wer_oracle, wer_max))

df_report = df_check[['asr_trans_stc','std_trans_wec','batch_num']].groupby(by=['batch_num']).sum()
df_report['wer'] = df_report['std_trans_wec'] / df_report['asr_trans_stc']
df_report.loc['All'] = [df_report['asr_trans_stc'].sum(), df_report['std_trans_wec'].sum(), df_report['std_trans_wec'].sum() / df_report['asr_trans_stc'].sum()] 


# Predict WER and Make Decision

df_check['wer'] = df_check['std_trans_wec'] / df_check['asr_trans_stc']
df_check['wec'] = df_check['std_trans_wec'] - df_check['num_q_mark']

relabeling = 0.1
threshold = max(0.1, np.quantile(df_check['wer'], 1-relabeling))
need_relabeling = (df_check['wer'] > threshold) & ((df_check['wec'] > 2) | (df_check['num_q_mark'] > df_check['asr_trans_stc'] * 0.5))
print('wer threshold = {:.2f}, need relabeling: count = {:d}, percent = {:.2f}'.format(threshold, sum(need_relabeling), sum(need_relabeling)/len(df_check)))

df_check['Reject'][need_relabeling] = df_check.apply(lambda x: reject_reason(x, x['batch_num']), axis=1)
df_check['Approve'][df_check['Reject'].isnull()] = 'x'

# Verify
if process == 'process':
    cols = ['Input.audio_url','Approve','Reject','std_trans','asr_trans','std_trans_wec','wer']
    df8 = df_check[need_relabeling].reset_index(drop=True)
    i = 0
    for col in cols:
        if i < len(df8):
            print('{:s} = {:s}'.format(col, str(df8[col][i])))

            
# Generate CSV file and upload to MTurk

cols = ['HITId','WorkerId','Approve','Reject']
all_reject_cnt = 0
for i, batch_num in enumerate(batch_nums):
    df_submit = df_submits[i]
    df_out = df_submit[ori_cols].drop(columns=['Approve','Reject']).merge(df_check[cols], on = ['HITId','WorkerId'])
    df_out.to_csv(os.path.join(folder, 'LibriCrowd', 'batch_result/{:s}/{:d}_out.csv'.format(process, batch_num)), index=False)
    n_reject = sum(~df_out['Reject'].isnull())
    rep_rate  = n_reject / len(df_out)
    all_reject_cnt += n_reject
    print('batch_num = {:04d}, n_reject = {:02d}, rep_rate = {:.2f}'.format(batch_num, n_reject, rep_rate))

print('batch_num = all, n_reject = {:02d}, rep_rate = {:.2f}'.format(all_reject_cnt, all_reject_cnt/len(df_check)))
