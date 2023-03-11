import os, time, glob, json
import hashlib
from typing import List, Tuple

import utils 
import requests
from threading import Thread
import time

from werkzeug.datastructures import FileStorage 

import torch
import torchaudio
from torch import Tensor

from flask import Flask, request, render_template, jsonify




global model 

model = None
results_dictionary = dict()

app = Flask(__name__)


def hash_string(string: str) -> int:
    return int(hashlib.sha1(string.encode("utf-8")).hexdigest(), 16) % (10 ** 8)


def convert_to_wav(
        input_path: str, 
        output_path: str,
        sample_rate: int = 16000
    ) -> None:

    cmd = "ffmpeg -y -i {} -v quiet -c:a pcm_s16le -ac 1 -ar {} {}".format(input_path, sample_rate, output_path)
    os.system(cmd)


def split_audio(
        wav_input_path: str,
        segment_time: float,
        output_folder: str = 'temp'
    ) -> List[str]:

    os.system("mkdir -p {}".format(output_folder))
    cmd = "ffmpeg -y -i {} -v quiet -f segment -segment_time {} -c copy {}/output%03d.wav".format(wav_input_path, segment_time, output_folder)
    os.system(cmd)

    segment_paths = glob.glob(os.path.join(output_folder, "*.wav"))
    segment_paths = sorted(segment_paths)
    print("split_audio done: ", len(segment_paths), "segments")
    return segment_paths

def normalize_audio(
            dns_home: str,
            file_request: FileStorage
        ) -> Tuple[str, str]:

    save_path = os.path.join(dns_home,'static/upload') + file_request.filename
    file_request.save(save_path)

    file_name = file_request.filename.split('.')[-2].replace(' ', '_')
    file_name_wav = "{}_nor.wav".format(file_name)
    file_wav_path_nor = os.path.join(dns_home, 'static/upload', file_name_wav)
    convert_to_wav(save_path, file_wav_path_nor)
    os.system("rm {}".format(save_path))

    return file_wav_path_nor, file_name

def normalize_audio2(
            dns_home: str,
            save_path: str
        ) -> Tuple[str, str]:

    # file_request.save(save_path)
    filename = save_path.split("/")[-1]
    file_name = filename.split('.')[-2].replace(' ', '_')
    file_name_wav = "{}_nor.wav".format(file_name)
    file_wav_path_nor = os.path.join(dns_home, 'static/upload', file_name_wav)
    convert_to_wav(save_path, file_wav_path_nor)
    # os.system("rm {}".format(save_path))

    return file_wav_path_nor, file_name


@app.route("/results", methods=["POST"])
def get_results():
    global results_dictionary
    data_output = {
            "status": "processing",
            "results": {
                "mixed_filepath": '',
                "speaker1_file": '',
                "speaker2_file": '',
                "speaker1_text": '',
                "speaker2_text": ''
            }
        }
    data = request.get_json()
    task_id = data['task_id']
    params = results_dictionary.get(task_id, None)
    if params:
        data_output["status"] = "done"
        data_output["results"] = params

    return jsonify(data_output)

def separation_long_audio(
            task_id: str,
            dns_home: str,
            save_path: str,
            current_time: int
        ) -> None:
    
    global results_dictionary
    file_request_wav_path_nor, file_request_name = normalize_audio2(dns_home, save_path)
    mix_name = file_request_name
    mixed_filepath = os.path.join(dns_home,'static/upload', '{}_{}.wav'.format(mix_name, current_time))
    os.system("mv {} {}".format(file_request_wav_path_nor, mixed_filepath))

    wav_arrays = list()

    segment_paths = split_audio(mixed_filepath, segment_time= 6, output_folder= 'temp')

    for seg_p in segment_paths:
        temp_array = utils._process(seg_p, model)
        wav_arrays.append(temp_array)
    print("process done")
    separated = torch.cat(wav_arrays, dim= 1)
    os.system("rm temp/*.wav")

    data = get_params(dns_home, mix_name, current_time, separated, mixed_filepath)

    results_dictionary[task_id] = data

    return 


def get_params(
        dns_home: str,
        mix_name: str,
        current_time: int,
        separated: Tensor,
        mixed_filepath: str,
    ) -> dict:

    out_file1_path = os.path.join(dns_home,'static/upload', mix_name + '_speaker1_{}.wav'.format(current_time))
    out_file2_path = os.path.join(dns_home,'static/upload', mix_name + '_speaker2_{}.wav'.format(current_time))

    torchaudio.save(out_file1_path, separated[:, :, 0], sample_rate= 16000)
    torchaudio.save(out_file2_path, separated[:, :, 1], sample_rate= 16000)

    mix_url = '/'.join(mixed_filepath.split('/')[-3:])
    spk1_url = '/'.join(out_file1_path.split('/')[-3:])
    spk2_url = '/'.join(out_file2_path.split('/')[-3:])
    # spk1_url = out_file1_path
    # spk2_url = out_file2_path

    data = {
                "mixed_filepath": mix_url,
                "speaker1_file": spk1_url,
                "speaker2_file": spk2_url,
            }
    
    return data


@app.route("/", methods=["GET","POST"])
def index():
    dns_home = "/home/hieule/speech-separation/main"
    current_time = int(time.time())

    if request.method == "GET":
        print("get")
        return render_template("index.html")
    else:
        print("post")
        if len(request.files) == 2:
            file1 = request.files["file1"]
            file2 = request.files["file2"]

            file1_wav_path_nor, file1_name = normalize_audio(dns_home, file1)
            file2_wav_path_nor, file2_name = normalize_audio(dns_home, file2)

            mix = utils.prepare_mixed(file1_wav_path_nor, file2_wav_path_nor)

            mix_name = file1_name + '_' + file2_name
            mixed_filepath = os.path.join('static/upload', '{}_{}.wav'.format(mix_name, current_time))
            mix = torch.tensor(mix)
            mix.to('cpu')
            torchaudio.save(mixed_filepath, mix.unsqueeze(0), sample_rate= 16000)
            separated = utils._process(mixed_filepath, model)
            print("process done")

            data = get_params(dns_home, mix_name, current_time, separated, mixed_filepath)

            return render_template( "index.html", **data)

        elif len(request.files) == 1:
            if request.files.get("file"):
                file3 = request.files["file"]
            elif request.files.get("file3"):
                file3 = request.files["file3"]
            elif request.files.get("file4"):
                file3 = request.files["file4"]
            else:
                print("file not found on request body!")
            file3_name = file3.filename
            print(file3)
            
            task_id = hash_string(file3_name)
            task_id = "{}_{}".format(task_id, current_time)
            save_path = os.path.join(dns_home, 'static/upload', 'auth_' + task_id + file3_name)
            mixed_filepath = os.path.join('static/upload', 'auth_' + task_id + file3_name)
            
            file3.save(save_path)
            print(f"{save_path} is exists:", os.path.exists(save_path))
            t = Thread(target=separation_long_audio, args=(task_id, dns_home, save_path, current_time, ))
            t.start()

            output = {
                "task_id": task_id,
                "status": 200,
                "results": {
                    "record_filepath": mixed_filepath,
                }
            }

            return jsonify(output)

if __name__ == "__main__":
    print("App run!")

	#load model
    port = 8080
    host = '0.0.0.0'
    model = utils._load_model()
    print("load model done!")
    app.run(debug=True, port = port, host=host, ssl_context="adhoc")