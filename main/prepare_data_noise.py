import os
import csv
import glob
import random
import argparse
from typing import List


import audiofile as af
from pydub import AudioSegment



parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, required=True, default="/content/drive/MyDrive/SpeechSeparation/dataset/output_folder/save_13kdata/data_tr.csv")
parser.add_argument('--output_csv', type=str, required=True, default="/content/drive/MyDrive/SpeechSeparation/dataset/output_folder/save_13kdata/data_tr.csv")
parser.add_argument('--data_type', type=str, required=True, default="tr")
parser.add_argument('--noise_folder', type=str, required=True, default="/content/drive/MyDrive/SpeechSeparation/RIRS_NOISES/pointsource_noises")
parser.add_argument('--output_mix_noise_folder', type=str, required=True, default="wav16k_tr_13k")

# python3.7 prepare_data_noise.py --input_csv ../../dataset/output_folder/save_v2/data_tt.csv --output_csv ../../dataset/output_folder/save_v2/data_tt_noise.csv --data_type tt --noise_folder ../../dataset/noise --output_mix_noise_folder wav16k_tt_noise


def match_target_amplitude(
            sound: AudioSegment, 
            target_dBFS: float = -30
        ) -> AudioSegment:
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def load_noise(
        path: str,
        db_range: list = [-40,-20],
    ) -> AudioSegment:

    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(16000)

    a,b = db_range
    target_dBFS = random.randint(a,b)
    audio = match_target_amplitude(audio, target_dBFS= target_dBFS)
    return audio


def mix_two_file(
            wav1: AudioSegment,
            wav2: AudioSegment,
            output_path: str
        ) -> None: 
    
    output = wav1.overlay(wav2, loop= True)
    output.export(output_path, format='wav', bitrate='512k', parameters=["-ac", "1", "-ar", "16000", "-acodec", "pcm_f32le"])


def load_noise_paths(noise_folder: str) -> list:
    noise_paths = glob.glob(os.path.join(noise_folder, "*.wav"))
    return noise_paths



def save_csv(
        csv_columns: list,
        save_path: str,
        row_data: List[dict],
    ) -> None:

    with open(save_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        for row in row_data:
            writer.writerow(row)



if __name__ == "__main__":
    args_input = parser.parse_args()

    input_csv = args_input.input_csv
    output_csv = args_input.output_csv
    data_type = args_input.data_type
    noise_folder = args_input.noise_folder
    output_mix_noise_folder = args_input.output_mix_noise_folder

    noise_paths = load_noise_paths(noise_folder)
    print(len(noise_paths))
    noise_paths = [path for path in noise_paths if af.duration(path) < 20]

    k = 0
    temp = list()
    name_dataset = "wav16k_{}".format(data_type)

    csv_columns = [
        "ID",
        "duration",
        "mix_wav",
        "mix_wav_format",
        "mix_wav_opts",
        "s1_wav",
        "s1_wav_format",
        "s1_wav_opts",
        "s2_wav",
        "s2_wav_format",
        "s2_wav_opts",
    ]

    with open(input_csv) as fp:
        for line in fp:
            
            if k == 0: 
                k += 1
                continue
            
            line = line.strip()
            ID,duration,mix_wav,mix_wav_format,mix_wav_opts,s1_wav,s1_wav_format,s1_wav_opts,s2_wav,s2_wav_format,s2_wav_opts = line.split(',')

            random_noise_path = random.sample(noise_paths, k= 1)[0]            
            noise_segment = load_noise(random_noise_path)
            mix_wav_segment = AudioSegment.from_file(mix_wav)

            new_mix_path = mix_wav.replace(name_dataset, output_mix_noise_folder)
            new_dir = os.path.dirname(new_mix_path)
            os.system("mkdir -p {}".format(new_dir))

            mix_two_file(mix_wav_segment, noise_segment, output_path= new_mix_path)

            row = {
                "ID": ID,
                "duration": duration,
                "mix_wav": new_mix_path,
                "mix_wav_format": "wav",
                "mix_wav_opts": None,
                "s1_wav": s1_wav,
                "s1_wav_format": "wav",
                "s1_wav_opts": None,
                "s2_wav": s2_wav,
                "s2_wav_format": "wav",
                "s2_wav_opts": None,
            }

            temp.append(row)

    save_csv(csv_columns, save_path= output_csv, row_data= temp)
