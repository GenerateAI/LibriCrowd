# LibriCrowd Analyze Result

__author__ = 'jg'

import jiwer
import pandas as pd

dataset_name = 'train_other_60h'
df = pd.read_csv('./data/LibriCrowd/libricrowd_{:s}_crowd_1.csv'.format(dataset_name), keep_default_na=False)
d_rename = {'utt':'utt_id', 'gold_trans':'trans_ground_truth'}
for col in df.columns:
    if 'std_trans' in col:
        d_rename[col] = col.replace('std_trans','trans_human')
df.rename(columns = d_rename, inplace=True)
df.info()
df.to_csv('./data/raw/{:s}_raw.csv'.format(dataset_name.replace('_','-')), index=False)

def compute_metrics(gold_trans, std_trans):
    metrics = jiwer.compute_measures(gold_trans, std_trans)
    return metrics

def error_type(df, gold_trans='gold_trans', std_trans='std_trans'):
    df[std_trans] = df[std_trans].fillna('')
    df['std_trans_metrics'] = df.apply(lambda x: compute_metrics(x[gold_trans], x[std_trans]), axis=1)
    df['gold_trans_stc'] = df[gold_trans].apply(lambda x: len(x.split()))
    df['std_trans_stc'] = df[std_trans].apply(lambda x: len(x.split()))
    df['std_trans_del'] = df['std_trans_metrics'].apply(lambda x: x['deletions'])
    df['std_trans_ins'] = df['std_trans_metrics'].apply(lambda x: x['insertions'])
    df['std_trans_sub'] = df['std_trans_metrics'].apply(lambda x: x['substitutions'])
    df['std_trans_wec'] = df['std_trans_metrics'].apply(lambda x: x['deletions'] + x['insertions'] + x['substitutions'])
    res = dict()
    res['CNT'] = len(df)
    res['LEN'] = df['std_trans_stc'].sum() / len(df)
    res['DEL'] = df['std_trans_del'].sum() / df['gold_trans_stc'].sum() * 100
    res['INS'] = df['std_trans_ins'].sum() / df['gold_trans_stc'].sum() * 100
    res['SUB'] = df['std_trans_sub'].sum() / df['gold_trans_stc'].sum() * 100
    res['WER'] = df['std_trans_wec'].sum() / df['gold_trans_stc'].sum() * 100
    return res

dataset = 'test-other'

df_check = pd.read_csv('./data/raw/{:s}_raw.csv'.format(dataset))
df_check['trans_ground_truth'][df_check['trans_ground_truth']==' '] = 'ImputedNA'
df_check.info()

res = error_type(df_check, gold_trans='trans_ground_truth', std_trans='trans_human_crowd')
print('dataset = {:s}, col = {:s}: count = {:d}, len = {:.1f}, del = {:.2f}%, ins = {:5.2f}%, sub = {:5.2f}%, wer = {:5.2f}%'\
      .format(dataset, col, res['CNT'], res['LEN'], res['DEL'], res['INS'], res['SUB'], res['WER']))


