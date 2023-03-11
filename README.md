Hướng dẫn chạy thực nghiệm Đồ án Ứng dụng và đánh giá mô hình SepFormer cho bài toán phân tách giọng nói
========================================



## Requirements

```bash

python version >= 3.7

sudo apt-get install -y gnuplot-x11
sudo apt-get install -y gnuplot
sudo apt install octave ffmpeg

```

```bash

pip install requirement.txt
pip install extra-dependencies.txt
```


## Data Preprocess

#### Step 1: Prepare struct project


```bash

|___/root_folder
    |___SpeechSeparation
        |___main
        |    |___config
        |    |___hparams
        |    |___meta
        |    |___speechbrain
        |    |___static
        |    |___templates
        |    |___conv_stft.py
        |    |___make_log.py
        |    |___prepare_data_noise.py
        |    |___prepare_data.py
        |    |___eval_sdr.py
        |    |___server.py
        |    |___train_STFT.py
        |    |___train.py
        |    |___utils.py
        |___README.md
        |___env_requirement.txt
        |___extra-dependencies.txt
        |___lint-requirements.txt
        |___requirement.txt
        |___RIRS_NOISES
            |___pointsource_noises
            |        |___noise-free-sound-0000.wav
            |        |___noise-free-sound-0001.wav
            |        ...
            |    ...
        |___dataset
            |___data_tr
            |    |___train_spk1
            |        |___train_spk1_0.wav
            |        |___train_spk1_1.wav
            |        ...
            |    |___train_spk2
            |        |___train_spk2_0.wav
            |        |___train_spk2_1.wav
            |        ...
            |    ...
            |___data_vd
            |    |___dev_spk1
            |        |___dev_spk1_0.wav
            |        |___dev_spk1_1.wav
            |        ...
            |    |___dev_spk2
            |        |___dev_spk2_0.wav
            |        |___dev_spk2_1.wav
            |        ...
            |    ...
            |___output_folder
            |    |___save


```


#### Step 2: Make log input

```bash

cd /root_folder/SpeechSeparation/main

python make_log.py \
    --folder_dataset /root_folder/SpeechSeparation/dataset \
    --log_path_train /root_folder/SpeechSeparation/dataset/mix_2_spk_tr.txt \
    --log_path_valid /root_folder/SpeechSeparation/dataset/mix_2_spk_vd.txt \
    --max_mixture_audio_train 20000 \
    --max_mixture_audio_valid 600

```

#### Step 3: Prepare input training

```bash

cd /root_folder/SpeechSeparation/main

python prepare_data.py --data_type tr,vd

```

## Training Sepformer model

```bash

cd /root_folder/SpeechSeparation/main

python train.py hparams/sepformer.yaml


```


## API 

Download model and copy to `/root_folder/SpeechSeparation/dataset/output_folder/save`

```bash

cd /root_folder/SpeechSeparation/main

python server.py hparams/run.yaml


```