import os, re, string
import numpy as np
import pandas as pd


def audio2utt(audio_url):
    if '/' in audio_url:
        audio_url = audio_url.split('/')[-1]
    audio = audio_url.split('.')[0]
    book = audio.split('-')[0]
    chap = audio.split('-')[1]
    uttr = audio.split('-')[2]
    utt = book + '_' + chap + '_' + str(int(uttr))
    return utt


def normalize_word(word):
    word = strnum2word.get(word, word)
    if word in ('okay', 'ok'): return 'OK'
    if word in ('mr', 'mr.'): return 'Mister'
    if word in ('ms', 'ms.'): return 'Miss'
    if word == 'mrs.': return 'Misses'
    if word == 'ya': return 'you'
    if word in ('cos','coz'): return 'because'
    if word in ('ah','eh','hm','um','er','ahhh'): return ''
    if word == "there's": return 'there is'
    if word == "i'm": return 'i am'
    if word == "i'll": return 'i will'
    if word == "it's": return 'it is'
    if word == "she's": return 'she is'
    if word == "he's": return 'he is'
    if word == "where's": return 'where is'
    if word == "what's": return 'what is'
    if word == "when's": return 'when is'
    if word == "why's": return 'why is'
    if word == "who's": return 'who is'
    if word == "you're": return 'you are'
    if word == "you've": return 'you have'
    if word == "don't": return 'do not'
    if word == "didn't": return 'did not'
    if word == "couldn't": return 'could not'
    if word == "won't": return 'will not'
    if word == "wasn't": return 'was not'
    if word == "o'clock": return 'oclock'
    if word == "dwarf's": return 'dwarfs'
    if word == 'wanna': return 'want to'
    if word == 'k': return 'OK'
    if word == 'anytime': return 'any time'
    if word[-1] == '.' and word not in ('a.m.','p.m.'): return word[:-1]
    return word.lower()


def normalize_trans(trans):
    trans = trans.lower()
    trans = trans.replace('?','').replace(',','').replace('.','')
    words = trans.split()
    for i in range(len(words)):
        words[i] = normalize_word(words[i])
    return ' '.join(words)


def cost_edit_distance(source: list, target: list, insert_cost=3, delete_cost=3, replace_cost=4):
    """
    Given two lists of tokens, calculate weighted edit distance
    """
    m = len(source)
    n = len(target)
    if target == source:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m
    d = np.zeros((n + 1, m + 1))
    f = np.zeros((n + 1, m + 1))
    d[0, 0] = 0
    f[0, 0] = 0
    for i in range(1, n + 1):
        d[i, 0] = d[i - 1, 0] + insert_cost
        f[i, 0] = f[i - 1, 0] + 1
    for j in range(1, m + 1):
        d[0, j] = d[0, j - 1] + delete_cost
        f[0, j] = f[0, j - 1] + 1
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            multiplier = int(not target[i - 1] == source[j - 1])
            d_costs = [d[i - 1, j] + insert_cost,
                       d[i - 1, j - 1] + multiplier * replace_cost,
                       d[i, j - 1] + delete_cost]
            f_costs = [f[i - 1, j] + 1,
                       f[i - 1, j - 1] + multiplier,
                       f[i, j - 1] + 1]
            idx = int(np.argmin(d_costs))
            assert isinstance(idx, int)
            d[i, j] = d_costs[idx]
            f[i, j] = f_costs[idx]
    return f[n, m]


def compute_wer(t1, t2):
    """
    Given two transcriptions, calculate Word Error Rate (WER)
    """
    word_error_count = cost_edit_distance(t1.split(), t2.split())
    return word_error_count


def reject_reason(df, batch_num):
    asr_trans, std_trans = df.asr_trans, df.std_trans
    asr_trans = asr_trans.replace("'", '')
    missing = (df.std_trans_stc < df.asr_trans_stc * 0.7)
    wec = df.std_trans_wec
    if missing:
        reason = 'the transcription is not correct, missing a lot (batch id: {:d}). your answer: {:s}, groundtruth: {:s}. #errors = {:.0f}'\
        .format(batch_num, std_trans, asr_trans, wec)
    else:
        reason = 'the transcription is not correct, many errors (batch id: {:d}). your answer: {:s}, groundtruth: {:s}. #errors = {:.0f}'\
        .format(batch_num, std_trans, asr_trans, wec)
    return reason