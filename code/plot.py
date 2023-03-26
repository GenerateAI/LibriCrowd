# Plot ASR models trained on LibriCrowd performance

__author__ = 'jg'

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5)

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y1 = [0, 1.18, 2, 4.32, 6.4, 7.98, 8.37, 10.61, 11.78, 12.91, 14.37] # werr_wo_lm
# y2  = [0, 4.22, 4.94, 6.31, 7.42, 10.27, 12.56, 14.47, 15.37, 14.47, 18.27] # werr_w_lm
y1 = [0, 2.68, 3.3, 5.87, 7.97, 8.20, 8.37, 12.24, 13.44, 14.58, 16.06] # werr_wo_lm
y2  = [0, 4.22, 5.53, 7.40, 8.53, 9.09, 12.56, 15.65, 15.37, 15.65, 19.49] # werr_w_lm

plt.axhline(y=5, color=(0, 0, 0, 0.2), linestyle='-')
plt.axhline(y=10, color=(0, 0, 0, 0.2), linestyle='-')
plt.axhline(y=15, color=(0, 0, 0, 0.2), linestyle='-')
# plt.axhline(y=10, color=(0, 0, 0, 0.2), linestyle='-')
plt.plot(x, y1, linestyle='-', marker='o', color='LightSkyBlue', label='Wav2Vec2 (w/ LM)', linewidth=3.0, markersize=10)
plt.plot(x, y2, linestyle='-', marker='v', color='GoldEnrod', label='Wav2Vec2 (w/o LM)', linewidth=3.0, markersize=10)
plt.axis([0, 11, 0, 20])
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.yticks(np.arange(min(y2), max(y2)+2, 5.0))
plt.xlabel('Transcription Label Quality: TWER (%)', fontsize=15)
plt.ylabel('ASR Model: R_WER (%)', fontsize=15)
plt.legend(fontsize=15, loc='upper left')
plt.savefig('./wer_twer.png')

x  = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80]
y1 = [9.87, 8.45, 7.81, 7.69, 7.4, 7.30, 7.2, 7.140, 7.11, 7.0, 6.96] # TWER = 5%
y2 = [9.52, 7.79, 7.09, 6.73, 6.0, 5.60, 5.39, 5.3, 5.29, 5.2, 5.11] # TWER = 0%

plt.axhline(y=5, color=(0, 0, 0, 0.2), linestyle='-')
plt.axhline(y=6, color=(0, 0, 0, 0.2), linestyle='-')
plt.axhline(y=7, color=(0, 0, 0, 0.2), linestyle='-')
plt.axhline(y=8, color=(0, 0, 0, 0.2), linestyle='-')
plt.axhline(y=9, color=(0, 0, 0, 0.2), linestyle='-')
plt.plot(x, y1, linestyle='-', marker='v', color='GoldEnrod', label='WavLM (TWER = 5%)', linewidth=3.0, markersize=10)
plt.plot(x, y2, linestyle='-', marker='o', color='LightSkyBlue', label='WavLM (TWER = 0%)', linewidth=3.0, markersize=10)
plt.axis([0, 81, 4.8, 10])
plt.xticks([1, 5, 10, 20, 30, 40, 50, 60, 70, 80])
plt.yticks([5, 6, 7, 8, 9, 10])
plt.xlabel('Training Data Size (hour)', fontsize=15)
plt.ylabel('ASR Model: WER (%)', fontsize=15)
plt.legend(fontsize=15, loc='upper right')
plt.savefig('./wer_datasize.png')
