# LibriCrowd Error Correction and Evaluation

__author__ = 'jg'


import datetime, os, re, string
import numpy as np
import pandas as pd
import jiwer
from ecm.agr import normalize
from ecm.ecm import ROVER
from utils import cost_edit_distance, compute_wer, normalize_trans, normalize_word


folder = '../data/LibriCrowd'
dataset = 'train-mixed-10h' # test-other, dev-other, test-clean, dev-clean, train-mixed-10h, train-other-10h
goldpath = os.path.join(folder, 'libricrowd_{:s}_trans.csv'.format(dataset.replace('-','_')))
crowdpath = os.path.join(folder, 'libricrowd_{:s}_crowd.csv'.format(dataset.replace('-','_')))
crowdpath_5 = os.path.join(folder, 'libricrowd_{:s}_crowd_5.csv'.format(dataset.replace('-','_')))
crowdpath_raw = os.path.join(folder, 'libricrowd_{:s}_crowd_raw.csv'.format(dataset.replace('-','_')))

# Load Ground Truth and MSR

def find_csv_filenames(path_to_dir, suffix='.csv'):
    filenames = os.listdir(path_to_dir)
    filepaths = [os.path.join(path_to_dir, filename) for filename in filenames if filename.endswith(suffix)]
    return filepaths

def audio2utt(audio_url):
    audio = audio_url.split('.')[0]
    book = audio.split('-')[0]
    chap = audio.split('-')[1]
    try:
        uttr = audio.split('-')[2]
    except:
        print(audio_url)
    utt = book + '_' + chap + '_' + str(int(uttr))
    return utt


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
    

def SubmitTime2DateTime(s):
    '''
    s = 'Wed Dec 28 17:35:02 PST 2022'
    t = '2022-12-28 17:35:02'
    '''
    e = '%a %b %d %H:%M:%S %Z %Y'
    d = datetime.datetime.strptime(s, e)
    t = datetime.datetime.strftime(d, '%Y-%m-%d %H:%M:%S')
    return t

df_gt = pd.read_csv(goldpath)
df_gt.info()

filepaths = find_csv_filenames(os.path.join(folder, dataset))
df_submits = []
for filepath in filepaths:
    df_submits.append(pd.read_csv(filepath))
df_submit = pd.concat(df_submits)
df_submit.dropna(subset=['Input.audio_url'], inplace=True)
df_submit = df_submit[df_submit['Input.audio_url'].apply(lambda x:'.txt' not in x)].reset_index(drop=True)
df_submit['taskId'] = df_submit['HITId'] + '|' + df_submit['WorkerId']

# merge std with gold and rename columns
df_submit['utt'] = df_submit['Input.audio_url'].apply(lambda x: audio2utt(x))
df_all = df_submit.merge(df_gt[['utt','gt_trans']], on = ['utt'], how = 'left')
df_all = df_all.rename(columns={'Answer.transcription':'msr_trans'})
df_all['msr_trans'] = df_all['msr_trans'].fillna('ImputedNA')
df_all['num_q_mark'] = df_all['msr_trans'].apply(lambda x: x.count('?'))
df_all['submittime'] = df_all['SubmitTime'].apply(lambda x: SubmitTime2DateTime(x))
# normalize trans
df_all['gt_trans'] = df_all['gt_trans'].apply(lambda x: normalize_trans(x))
df_all['msr_trans'] = df_all['msr_trans'].apply(lambda x: normalize_trans(x))
df_all.info()

def get_wer(df, msr_trans='msr_trans', reuse=True):
    if reuse == False or 'msr_trans_wec' not in df.columns:
        print('compute msr_trans_wec ...', end='\r')
        df['gt_trans_stc'] = df['gt_trans'].apply(lambda x: len(x.split()))
        df['msr_trans_stc']  = df[msr_trans].apply(lambda x: len(x.split()))
        df['msr_trans_wec']  = df.apply(lambda x: compute_wer(x.gt_trans, x[msr_trans]), axis=1)
    df1 = df.groupby('utt').agg({'msr_trans_wec': ['mean', 'min', 'max']})
    df2 = df.groupby('utt').agg({'gt_trans_stc': ['min']})
    wer_avg = df1[('msr_trans_wec', 'mean')].sum() / df2[('gt_trans_stc', 'min')].sum()
    wer_min = df1[('msr_trans_wec', 'min')].sum()  / df2[('gt_trans_stc', 'min')].sum()
    wer_max = df1[('msr_trans_wec', 'max')].sum()  / df2[('gt_trans_stc', 'min')].sum()
    return wer_avg, wer_min, wer_max

def compute_metrics(gt_trans, msr_trans):
    metrics = jiwer.compute_measures(gt_trans, msr_trans)
    return metrics


df_all['gt_trans_stc'] = df_all['gt_trans'].apply(lambda x: len(x.split()))
df_all['msr_trans_stc']  = df_all['msr_trans'].apply(lambda x: len(x.split()))
df_all['msr_trans_wec']  = df_all.apply(lambda x: compute_wer(x.gt_trans, x.msr_trans), axis=1)

df_approved = df_all[df_all['AssignmentStatus'] == 'Approved'].reset_index(drop=True)
df_rejected = df_all[df_all['AssignmentStatus'] == 'Rejected'].reset_index(drop=True)
wer_avg_approve, wer_min_approve, wer_max_approve = get_wer(df_approved)
wer_avg_reject,  wer_min_reject,  wer_max_reject  = get_wer(df_rejected)
print('Approved: count = {:5d}, wer_avg = {:.4f}, wer_min = {:.4f}, wer_max = {:.4f}'.format(len(df_approved), wer_avg_approve, wer_min_approve, wer_max_approve))
print('Rejected: count = {:5d}, wer_avg = {:.4f}, wer_min = {:.4f}, wer_max = {:.4f}'.format(len(df_rejected), wer_avg_reject, wer_min_reject, wer_max_reject))


# Error Type Statistics for Approved and Rejected Transcriptions

def error_type(df, msr_trans='msr_trans'):
    df[msr_trans] = df[msr_trans].fillna('')
    df['msr_trans_metrics'] = df.apply(lambda x: compute_metrics(x.gt_trans, x[msr_trans]), axis=1)
    df['gt_trans_stc'] = df['gt_trans'].apply(lambda x: len(x.split()))
    df['msr_trans_stc'] = df[msr_trans].apply(lambda x: len(x.split()))
    df['msr_trans_del'] = df['msr_trans_metrics'].apply(lambda x: x['deletions'])
    df['msr_trans_ins'] = df['msr_trans_metrics'].apply(lambda x: x['insertions'])
    df['msr_trans_sub'] = df['msr_trans_metrics'].apply(lambda x: x['substitutions'])
    df['msr_trans_wec'] = df['msr_trans_metrics'].apply(lambda x: x['deletions'] + x['insertions'] + x['substitutions'])
    res = dict()
    res['CNT'] = len(df)
    res['LEN'] = df['msr_trans_stc'].sum() / len(df)
    res['DEL'] = df['msr_trans_del'].sum() / df['gt_trans_stc'].sum() * 100
    res['INS'] = df['msr_trans_ins'].sum() / df['gt_trans_stc'].sum() * 100
    res['SUB'] = df['msr_trans_sub'].sum() / df['gt_trans_stc'].sum() * 100
    res['WER'] = df['msr_trans_wec'].sum() / df['gt_trans_stc'].sum() * 100
    return res


res = error_type(df_approved)
print('dataset = {:s}, approved: count = {:d}, len = {:.1f}, del = {:.2f}%, ins = {:.2f}%, sub = {:.2f}%, wer = {:.2f}%'\
      .format(dataset, res['CNT'], res['LEN'], res['DEL'], res['INS'], res['SUB'], res['WER']))
res = error_type(df_rejected)
print('dataset = {:s}, rejected: count = {:d}, len = {:.1f}, del = {:.2f}%, ins = {:.2f}%, sub = {:.2f}%, wer = {:.2f}%'\
      .format(dataset, res['CNT'], res['LEN'], res['DEL'], res['INS'], res['SUB'], res['WER']))


# Save Data

cols = ['HITId','HITTypeId','Title','Description','Keywords','Reward','CreationTime','MaxAssignments','RequesterAnnotation','AssignmentDurationInSeconds','AutoApprovalDelayInSeconds','Expiration','AssignmentId','WorkerId','AssignmentStatus','AcceptTime','SubmitTime','AutoApprovalTime','ApprovalTime','RejectionTime','RequesterFeedback','WorkTimeInSeconds','LifetimeApprovalRate','Last30DaysApprovalRate','Last7DaysApprovalRate','Input.audio_url','msr_trans','utt','gt_trans','num_q_mark','submittime','gt_trans_stc','msr_trans_stc','msr_trans_wec','taskId']
df_crowd_raw = df_all[cols].reset_index(drop=True)
df_crowd_raw.to_csv(crowdpath_raw, index=False)


# Evaluate Relabeling

cols = ['utt','submittime','AssignmentId']
df = df_all[cols].sort_values(by=['submittime','utt']).reset_index(drop=True)

df1 = df.groupby('utt', as_index=False).apply(lambda x: x if len(x)==1 else x.iloc[[0]]).reset_index(level=0, drop=True)
df2 = df.groupby('utt', as_index=False).apply(lambda x: x if len(x)==1 else x.iloc[[1]]).reset_index(level=0, drop=True)
df3 = df.groupby('utt', as_index=False).apply(lambda x: x if len(x)==1 else x.iloc[[2]]).reset_index(level=0, drop=True)
df4 = df.groupby('utt', as_index=False).apply(lambda x: x if len(x)==1 else x.iloc[[3]]).reset_index(level=0, drop=True)
df5 = df.groupby('utt', as_index=False).apply(lambda x: x if len(x)==1 else x.iloc[[4]]).reset_index(level=0, drop=True)

df_5 = pd.concat([df1, df2, df3, df4, df5])
df_before_relabel = df_all.merge(df_5[['AssignmentId']], on = ['AssignmentId'], how='right')
df_after_relabel  = df_approved.copy()

wer_avg_before, wer_min_before, wer_max_before = get_wer(df_before_relabel)
wer_avg_after, wer_min_after, wer_max_after = get_wer(df_after_relabel)
print('Before Relabeling: wer_avg = {:.4f}, wer_min = {:.4f}, wer_max = {:.4f}'.format(wer_avg_before, wer_min_before, wer_max_before))
print('After  Relabeling: wer_avg = {:.4f}, wer_min = {:.4f}, wer_max = {:.4f}'.format(wer_avg_after, wer_min_after, wer_max_after))
relabel_rate = len(df_rejected) / len(df_approved)
num_worker_before = len(set(df_before_relabel['WorkerId']))
num_worker_after  = len(set(df_all['WorkerId']))
print('Relabeling percent = {:.2f}%, WER improve = {:.0f} bps'.format(100*relabel_rate, 1e4*(wer_avg_before - wer_avg_after)))
print('Number of workers: before = {:d}, after = {:d}'.format(num_worker_before, num_worker_after))


# ECMs

# Random and Oracle

fn1 = lambda x: x.loc[np.random.choice(x.index, size=1, replace=False) if len(x) >= 4 else np.random.choice(x.index, 4, True),:]
fn2 = lambda x: x.loc[np.random.choice(x.index, size=2, replace=False) if len(x) >= 4 else np.random.choice(x.index, 4, True),:]
fn3 = lambda x: x.loc[np.random.choice(x.index, size=3, replace=False) if len(x) >= 4 else np.random.choice(x.index, 4, True),:]
fn4 = lambda x: x.loc[np.random.choice(x.index, size=4, replace=False) if len(x) >= 4 else np.random.choice(x.index, 4, True),:]

df_before_random_1 = df_before_relabel.groupby('utt', as_index=False).apply(fn1).reset_index(drop=True)
df_before_random_2 = df_before_relabel.groupby('utt', as_index=False).apply(fn2).reset_index(drop=True)
df_before_random_3 = df_before_relabel.groupby('utt', as_index=False).apply(fn3).reset_index(drop=True)
df_before_random_4 = df_before_relabel.groupby('utt', as_index=False).apply(fn4).reset_index(drop=True)
df_after_random_1  = df_after_relabel.groupby('utt', as_index=False).apply(fn1).reset_index(drop=True)
df_after_random_2  = df_after_relabel.groupby('utt', as_index=False).apply(fn2).reset_index(drop=True)
df_after_random_3  = df_after_relabel.groupby('utt', as_index=False).apply(fn3).reset_index(drop=True)
df_after_random_4  = df_after_relabel.groupby('utt', as_index=False).apply(fn4).reset_index(drop=True)

wer_random_before, _, _  = get_wer(df_before_random_1)
wer_random_after, _, _   = get_wer(df_after_random_1)
_, wer_oracle2_before, _ = get_wer(df_before_random_2)
_, wer_oracle2_after, _  = get_wer(df_after_random_2)
_, wer_oracle3_before, _ = get_wer(df_before_random_3)
_, wer_oracle3_after, _  = get_wer(df_after_random_3)
_, wer_oracle4_before, _ = get_wer(df_before_random_4)
_, wer_oracle4_after, _  = get_wer(df_after_random_4)
wer_oracle5_before       = wer_min_before
wer_oracle5_after        = wer_min_after

print('wer_random1_before = {:.4f}, wer_random1_after = {:.4f}'.format(wer_random_before, wer_random_after))
print('wer_oracle2_before = {:.4f}, wer_oracle2_after = {:.4f}'.format(wer_oracle2_before, wer_oracle2_after))
print('wer_oracle3_before = {:.4f}, wer_oracle3_after = {:.4f}'.format(wer_oracle3_before, wer_oracle3_after))
print('wer_oracle4_before = {:.4f}, wer_oracle4_after = {:.4f}'.format(wer_oracle4_before, wer_oracle4_after))
print('wer_oracle5_before = {:.4f}, wer_oracle5_after = {:.4f}'.format(wer_oracle5_before, wer_oracle5_after))

# Longest

def f_longest(x):
    ref = -1
    res = -1
    for idx, msr_trans_stc in zip(x.index, x.msr_trans_stc):
        if msr_trans_stc > ref:
            ref = msr_trans_stc
            res = idx
    return x.loc[res,:]

df_longest_before = df_before_relabel.groupby('utt', as_index=False).apply(f_longest).reset_index(drop=True)
df_longest_after  = df_after_relabel.groupby('utt', as_index=False).apply(f_longest).reset_index(drop=True)

# Best Worker

df_before_relabel['approve_rate'] = df_before_relabel['LifetimeApprovalRate'].apply(lambda x: int(x.split('%')[0])/100)
df_after_relabel['approve_rate']  = df_after_relabel['LifetimeApprovalRate'].apply(lambda x: int(x.split('%')[0])/100)

def f_highest(x):
    ref = -1
    res = -1
    for idx, approve_rate in zip(x.index, x.approve_rate):
        if approve_rate > ref:
            ref = approve_rate
            res = idx
    return x.loc[res,:]

df_highest_before = df_before_relabel.groupby('utt', as_index=False).apply(f_highest).reset_index(drop=True)
df_highest_after  = df_after_relabel.groupby('utt', as_index=False).apply(f_highest).reset_index(drop=True)

# Oracle Worst

def f_best(x):
    ref = 9999
    res = -1
    for idx, wec in zip(x.index, x.msr_trans_wec):
        if wec < ref:
            ref = wec
            res = idx
    return x.loc[res,:]

def f_worst(x):
    ref = -1
    res = -1
    for idx, wec in zip(x.index, x.msr_trans_wec):
        if wec > ref:
            ref = wec
            res = idx
    return x.loc[res,:]

df_best_before = df_before_relabel.groupby('utt', as_index=False).apply(f_best).reset_index(drop=True)
df_best_after  = df_after_relabel.groupby('utt', as_index=False).apply(f_best).reset_index(drop=True)
df_worst_before = df_before_relabel.groupby('utt', as_index=False).apply(f_worst).reset_index(drop=True)
df_worst_after  = df_after_relabel.groupby('utt', as_index=False).apply(f_worst).reset_index(drop=True)

# Auto Correction

rover_before = ROVER().fit_predict(df_before_relabel[['utt','msr_trans']].rename(columns={"utt": "task", "msr_trans": "output"}))
rover_after = ROVER().fit_predict(df_after_relabel[['utt','msr_trans']].rename(columns={"utt": "task", "msr_trans": "output"}))

df_eval_before = df_before_relabel.merge(rover_before.reset_index(), left_on='utt', right_on='task').groupby('utt').agg(lambda x: x.iloc[0]).reset_index()
df_eval_before.rename(columns={'output': 'msr_trans_rover'}, inplace=True)
df_eval_after = df_after_relabel.merge(rover_after.reset_index(), left_on='utt', right_on='task').groupby('utt').agg(lambda x: x.iloc[0]).reset_index()
df_eval_after.rename(columns={'output': 'msr_trans_rover'}, inplace=True)

# Print
res = error_type(df_before_relabel)
print('dataset = {:s}, raw_before   : count = {:d}, len = {:.1f}, del = {:.2f}%, ins = {:5.2f}%, sub = {:5.2f}%, wer = {:5.2f}%'\
      .format(dataset, res['CNT'], res['LEN'], res['DEL'], res['INS'], res['SUB'], res['WER']))
res = error_type(df_worst_before)
print('dataset = {:s}, worst_before  : count = {:d}, len = {:.1f}, del = {:.2f}%, ins = {:5.2f}%, sub = {:5.2f}%, wer = {:5.2f}%'\
      .format(dataset, res['CNT'], res['LEN'], res['DEL'], res['INS'], res['SUB'], res['WER']))
res = error_type(df_before_random_1)
print('dataset = {:s}, random_before : count = {:d}, len = {:.1f}, del = {:.2f}%, ins = {:5.2f}%, sub = {:5.2f}%, wer = {:5.2f}%'\
      .format(dataset, res['CNT'], res['LEN'], res['DEL'], res['INS'], res['SUB'], res['WER']))
res = error_type(df_longest_before)
print('dataset = {:s}, longest_before: count = {:d}, len = {:.1f}, del = {:.2f}%, ins = {:5.2f}%, sub = {:5.2f}%, wer = {:5.2f}%'\
      .format(dataset, res['CNT'], res['LEN'], res['DEL'], res['INS'], res['SUB'], res['WER']))
res = error_type(df_highest_before)
print('dataset = {:s}, highest_before: count = {:d}, len = {:.1f}, del = {:.2f}%, ins = {:5.2f}%, sub = {:5.2f}%, wer = {:5.2f}%'\
      .format(dataset, res['CNT'], res['LEN'], res['DEL'], res['INS'], res['SUB'], res['WER']))
res = error_type(df_eval_before, msr_trans='msr_trans_rover')
print('dataset = {:s}, correct_before: count = {:d}, len = {:.1f}, del = {:.2f}%, ins = {:5.2f}%, sub = {:5.2f}%, wer = {:5.2f}%'\
      .format(dataset, res['CNT'], res['LEN'], res['DEL'], res['INS'], res['SUB'], res['WER']))
res = error_type(df_best_before)
print('dataset = {:s}, best_before   : count = {:d}, len = {:.1f}, del = {:.2f}%, ins = {:5.2f}%, sub = {:5.2f}%, wer = {:5.2f}%'\
      .format(dataset, res['CNT'], res['LEN'], res['DEL'], res['INS'], res['SUB'], res['WER']))
print('-'*122)
res = error_type(df_after_relabel)
print('dataset = {:s}, raw_after    : count = {:d}, len = {:.1f}, del = {:.2f}%, ins = {:5.2f}%, sub = {:5.2f}%, wer = {:5.2f}%'\
      .format(dataset, res['CNT'], res['LEN'], res['DEL'], res['INS'], res['SUB'], res['WER']))
res = error_type(df_worst_after)
print('dataset = {:s}, worst_after   : count = {:d}, len = {:.1f}, del = {:.2f}%, ins = {:5.2f}%, sub = {:5.2f}%, wer = {:5.2f}%'\
      .format(dataset, res['CNT'], res['LEN'], res['DEL'], res['INS'], res['SUB'], res['WER']))
res = error_type(df_after_random_1)
print('dataset = {:s}, random_after  : count = {:d}, len = {:.1f}, del = {:.2f}%, ins = {:5.2f}%, sub = {:5.2f}%, wer = {:5.2f}%'\
      .format(dataset, res['CNT'], res['LEN'], res['DEL'], res['INS'], res['SUB'], res['WER']))
res = error_type(df_longest_after)
print('dataset = {:s}, longest_after : count = {:d}, len = {:.1f}, del = {:.2f}%, ins = {:5.2f}%, sub = {:5.2f}%, wer = {:5.2f}%'\
      .format(dataset, res['CNT'], res['LEN'], res['DEL'], res['INS'], res['SUB'], res['WER']))
res = error_type(df_highest_after)
print('dataset = {:s}, highest_after : count = {:d}, len = {:.1f}, del = {:.2f}%, ins = {:5.2f}%, sub = {:5.2f}%, wer = {:5.2f}%'\
      .format(dataset, res['CNT'], res['LEN'], res['DEL'], res['INS'], res['SUB'], res['WER']))
res = error_type(df_eval_after, msr_trans='msr_trans_rover')
print('dataset = {:s}, correct_after : count = {:d}, len = {:.1f}, del = {:.2f}%, ins = {:5.2f}%, sub = {:5.2f}%, wer = {:5.2f}%'\
      .format(dataset, res['CNT'], res['LEN'], res['DEL'], res['INS'], res['SUB'], res['WER']))
res = error_type(df_best_after)
print('dataset = {:s}, best_after    : count = {:d}, len = {:.1f}, del = {:.2f}%, ins = {:5.2f}%, sub = {:5.2f}%, wer = {:5.2f}%'\
      .format(dataset, res['CNT'], res['LEN'], res['DEL'], res['INS'], res['SUB'], res['WER']))

# Save data

df_crowd = df_gt.copy()
df_crowd['gt_trans'] = df_crowd['gt_trans'].apply(lambda x: normalize_trans(x))
df_crowd = df_crowd.rename(columns={'msr_trans':'msr_trans_synthetic'})
# random before
df_merge_random_before = df_before_random_1[['utt','msr_trans']].drop_duplicates(subset=['utt'])
df_crowd = df_crowd.merge(df_merge_random_before, on=['utt'], how='left')
df_crowd = df_crowd.rename(columns={'msr_trans':'msr_trans_random_before'})
# random after
df_merge_random_after = df_after_random_1[['utt','msr_trans']].drop_duplicates(subset=['utt'])
df_crowd = df_crowd.merge(df_merge_random_after, on=['utt'], how='left')
df_crowd = df_crowd.rename(columns={'msr_trans':'msr_trans_random_after'})
df_crowd['msr_trans_random_after'] = df_crowd['msr_trans_random_after'].fillna(df_crowd['msr_trans_random_before'])
# longest before
df_merge_longest_before = df_longest_before[['utt','msr_trans']].drop_duplicates(subset=['utt'])
df_crowd = df_crowd.merge(df_merge_longest_before, on=['utt'], how='left')
df_crowd = df_crowd.rename(columns={'msr_trans':'msr_trans_longest_before'})
df_crowd['msr_trans_longest_before'] = df_crowd['msr_trans_longest_before'].fillna(df_crowd['msr_trans_random_before'])
# longest after
df_merge_longest_after = df_longest_after[['utt','msr_trans']].drop_duplicates(subset=['utt'])
df_crowd = df_crowd.merge(df_merge_longest_after, on=['utt'], how='left')
df_crowd = df_crowd.rename(columns={'msr_trans':'msr_trans_longest_after'})
df_crowd['msr_trans_longest_after'] = df_crowd['msr_trans_longest_after'].fillna(df_crowd['msr_trans_random_after'])
# highest before
df_merge_highest_before = df_highest_before[['utt','msr_trans']].drop_duplicates(subset=['utt'])
df_crowd = df_crowd.merge(df_merge_highest_before, on=['utt'], how='left')
df_crowd = df_crowd.rename(columns={'msr_trans':'msr_trans_highest_before'})
df_crowd['msr_trans_highest_before'] = df_crowd['msr_trans_highest_before'].fillna(df_crowd['msr_trans_random_before'])
# highest after
df_merge_highest_after = df_highest_after[['utt','msr_trans']].drop_duplicates(subset=['utt'])
df_crowd = df_crowd.merge(df_merge_highest_after, on=['utt'], how='left')
df_crowd = df_crowd.rename(columns={'msr_trans':'msr_trans_highest_after'})
df_crowd['msr_trans_highest_after'] = df_crowd['msr_trans_highest_after'].fillna(df_crowd['msr_trans_random_after'])
# correct before
df_merge_correct_before = df_eval_before.reset_index()[['utt','msr_trans_rover']].drop_duplicates(subset=['utt','msr_trans_rover'])
df_crowd = df_crowd.merge(df_merge_correct_before, on=['utt'], how='left')
df_crowd['msr_trans_rover'] = df_crowd['msr_trans_rover'].fillna(df_crowd['msr_trans_random_before'])
df_crowd = df_crowd.rename(columns={'msr_trans_rover':'msr_trans_correct_before'})
# correct after
df_merge_correct_after = df_eval_after.reset_index()[['utt','msr_trans_rover']].drop_duplicates(subset=['utt','msr_trans_rover'])
df_crowd = df_crowd.merge(df_merge_correct_after, on=['utt'], how='left')
df_crowd['msr_trans_rover'] = df_crowd['msr_trans_rover'].fillna(df_crowd['msr_trans_random_after'])
df_crowd = df_crowd.rename(columns={'msr_trans_rover':'msr_trans_correct_after'})
# best before
df_merge_best_before = df_best_before[['utt','msr_trans']].drop_duplicates(subset=['utt'])
df_crowd = df_crowd.merge(df_merge_best_before, on=['utt'], how='left')
df_crowd = df_crowd.rename(columns={'msr_trans':'msr_trans_best_before'})
df_crowd['msr_trans_best_before'] = df_crowd['msr_trans_best_before'].fillna(df_crowd['msr_trans_random_before'])
# best after
df_merge_best_after = df_best_after[['utt','msr_trans']].drop_duplicates(subset=['utt'])
df_crowd = df_crowd.merge(df_merge_best_after, on=['utt'], how='left')
df_crowd = df_crowd.rename(columns={'msr_trans':'msr_trans_best_after'})
df_crowd['msr_trans_best_after'] = df_crowd['msr_trans_best_after'].fillna(df_crowd['msr_trans_random_after'])
# worst before
df_merge_worst_before = df_worst_before[['utt','msr_trans']].drop_duplicates(subset=['utt'])
df_crowd = df_crowd.merge(df_merge_worst_before, on=['utt'], how='left')
df_crowd = df_crowd.rename(columns={'msr_trans':'msr_trans_worst_before'})
df_crowd['msr_trans_worst_before'] = df_crowd['msr_trans_worst_before'].fillna(df_crowd['msr_trans_random_before'])
# worst after
df_merge_worst_after = df_worst_after[['utt','msr_trans']].drop_duplicates(subset=['utt'])
df_crowd = df_crowd.merge(df_merge_worst_after, on=['utt'], how='left')
df_crowd = df_crowd.rename(columns={'msr_trans':'msr_trans_worst_after'})
df_crowd['msr_trans_worst_after'] = df_crowd['msr_trans_worst_after'].fillna(df_crowd['msr_trans_random_after'])

df_crowd.info()
df_crowd.to_csv(crowdpath, index=False)

df_crowd_before = df_before_relabel[['utt','gt_trans','msr_trans','LifetimeApprovalRate','taskId']].rename(columns={'msr_trans':'msr_trans_crowd'})
df_crowd_before['relabel'] = 0
df_crowd_before['approve_rate'] = df_crowd_before['LifetimeApprovalRate'].apply(lambda x: int(x.split('%')[0])/100)
df_crowd_after  = df_after_relabel[['utt','gt_trans','msr_trans','LifetimeApprovalRate','taskId']].rename(columns={'msr_trans':'msr_trans_crowd'})
df_crowd_after['relabel'] = 1
df_crowd_after['approve_rate'] = df_crowd_before['LifetimeApprovalRate'].apply(lambda x: int(x.split('%')[0])/100)

df_crowd_before['len_gt_trans'] = df_crowd_before['gt_trans'].apply(lambda x:len(x.split()))
df1 = df_crowd_before.groupby(['utt'], as_index=False).nth(0).sort_values(by='len_gt_trans')
df2 = df_crowd_before.groupby(['utt'], as_index=False).nth(1).sort_values(by='len_gt_trans')
df3 = df_crowd_before.groupby(['utt'], as_index=False).nth(2).sort_values(by='len_gt_trans')
df4 = df_crowd_before.groupby(['utt'], as_index=False).nth(3).sort_values(by='len_gt_trans')
df5 = df_crowd_before.groupby(['utt'], as_index=False).nth(4).sort_values(by='len_gt_trans')
df_crowd_before = pd.concat([df1,df2,df3,df4,df5]).reset_index(drop=True)

df_crowd_after['len_gt_trans'] = df_crowd_after['gt_trans'].apply(lambda x:len(x.split()))
df1 = df_crowd_after.groupby(['utt'], as_index=False).nth(0).sort_values(by='len_gt_trans')
df2 = df_crowd_after.groupby(['utt'], as_index=False).nth(1).sort_values(by='len_gt_trans')
df3 = df_crowd_after.groupby(['utt'], as_index=False).nth(2).sort_values(by='len_gt_trans')
df4 = df_crowd_after.groupby(['utt'], as_index=False).nth(3).sort_values(by='len_gt_trans')
df5 = df_crowd_after.groupby(['utt'], as_index=False).nth(4).sort_values(by='len_gt_trans')
df_crowd_after = pd.concat([df1,df2,df3,df4,df5]).reset_index(drop=True)

cols = ['utt','taskId','gt_trans','msr_trans_crowd','approve_rate','relabel']
df_crowd_5 = pd.concat([df_crowd_before[cols], df_crowd_after[cols]]).reset_index(drop=True)
df_crowd_5.info()
df_crowd_5.to_csv(crowdpath_5, index=False)

