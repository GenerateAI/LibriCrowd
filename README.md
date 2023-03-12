# LibriCrowd
A large-scale crowdsourced English speech corpus with clean and noisy human transcriptions. 


## Dataset Summary
LibriCrowd is a corpus of approximately 100 hours scripted English speech with both clean and noisy human transcriptions. The raw audio files and ground truth transcriptions are selected from a subset of the well-known LibriSpeech corpus. More information about the dataset statistics and baseline models trained on it can be found in our Paper ["Human Transcription Quality Improvement"](https://amazon.awsapps.com/workdocs/index.html#/document/cde31c11ae43698bc2d1e41a7fb6e4f3bce3e8c544f94bc2dce1f313a8c73020)


## Supported Tasks
* Human Transcription Error Detection and Correction:

The dataset contains crowdsourced human transcriptions at various noisy levels. The transcription quality is measured by the Transcription Word Error Rate (TWER) of noisy human transcriptions and the ground truth reference. The task is to improve crowdsourced human transcription quality by developing Confidence Estimation Models (CEMs) to detect human errors, and Error Correction Models (ECMs) to refine human transcriptions. The CEM performance is evaluated by error prediction accuracy (precision, recall, F1) at word or utterance level. The ECM performance is evaluated by the TWER reduction between the raw and refined human transcripitons. 

* Robust Automatic Speech Recognition (ASR) System Evaluation:

The dataset can be used to finetune pretrained speech representations with the audio file and its human transcription text. The transcription has a clean version as well as a noisy version with a certain amount of human error. A robust ASR system is expect to have limited performance degradation when it's trained on nosiy data compared with the same model trained on clean data. The ASR system is evaluated by the Word Error Rate (WER %) of the ASR hypothesis compared to the ground truth reference. 

The controled noisy level (TWER %) can be obtained by randomly mixing the noisy human transcriptions with the ground truth reference. The robustness of ASR models is evaluated by using different levels of nosiy transcriptions as the training data, and then measure the WER. 


## Dataset Structure and Statistics

This dataset can be split into three subsets. For training, the data is split into 'train-other-10h', 'train-other-60h', 'train-mixed-10h'; For evaluation, the data is split into 'dev-clean', 'dev-other', 'test-clean', 'test-other', which is the same as in the LibriSpeech dev/test subsets. The entire dataset contains approximately 100 hours of English speech. Detailed statistic is listed below:

|      Subset      | # Utterances | speech hours | # Workers | # Responses |
|:----------------:|:------------:|:------------:|:---------:|:-----------:|
| train-other-10h  |         3165 |         10.0 |      1258 |       18673 |
| train-other-60h  |        17816 |         60.0 |      1136 |       20187 |
| train-mixed-10h  |         2763 |          9.8 |       616 |       14231 |
| dev-clean        |         2703 |          5.4 |       523 |       13994 |
| test-clean       |         2620 |          5.4 |       527 |       13587 |
| dev-other        |         2864 |          5.3 |       620 |       15235 |
| test-other       |         2939 |          5.1 |       989 |       15950 |
| all              |        34870 |        101.0 |      4433 |      111857 |


## Download
* The raw human transcriptions can be downloaded from ``./transcription/raw/``
* The processed human transcriptions can be downloaded from ``./transcription/processed/``
* The sample raw speech audio files can be viewed from ``./audio_sample/``. Full audio files can be downloaded:
    * [train-mixed-10h.tar.gz](https://www.dropbox.com/s/h86wodvi0f2qsdl/train-mixed-10h.tar.gz?dl=0)[598M]
    * [train-other-10h.tar.gz](https://www.dropbox.com/s/80eklq30r8gw078/train-other-10h.tar.gz?dl=0)[591M]
    * [train-other-60h.tar.gz](TBD)[3.7G]
    * [dev-clean.tar.gz](https://www.openslr.org/resources/12/dev-clean.tar.gz)[337M]
    * [dev-other.tar.gz](https://www.openslr.org/resources/12/dev-other.tar.gz)[314M]
    * [test-clean.tar.gz](https://www.openslr.org/resources/12/test-clean.tar.gz)[346M]
    * [test-other.tar.gz](https://www.openslr.org/resources/12/test-other.tar.gz)[328M]

## Licensing Information
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)


## Acknowledgements
* [LibriSpeech](https://www.openslr.org/12) dataset is used under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
* [Libri-Light](https://github.com/facebookresearch/libri-light) dataset is used under the the [MIT](https://opensource.org/license/mit/) license. 
* [LibriVox](https://librivox.org/) project is a free public domain audiobooks read by volunteers from around the world. All LibriVox Recordings are in the Public Domain and free to use without any restriction. 

