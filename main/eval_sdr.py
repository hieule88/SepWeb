import numpy as np
import utils 
import sys
import torch
from tqdm import tqdm
from speechbrain.nnet.losses import get_si_snr_with_pitwrapper


def sdr(references, estimates):
    # compute SDR for one song
    delta = 1e-7  # avoid numerical errors
    num = torch.sum(torch.square(references))
    den = torch.sum(torch.square(references - estimates))
    num += delta
    den += delta
    return 10 * torch.log10(num / den)

def count_sdr(input_csv, model):
    model.modules.eval()
    k=0
    num_sdr = 0.0
    sum_sdr = 0.0
    with open(input_csv) as fp:
        for line in tqdm(fp):
            if k == 0: 
                k += 1
                continue
            line = line.strip()
            ID,duration,mix_wav,mix_wav_format,mix_wav_opts,s1_wav,s1_wav_format,s1_wav_opts,s2_wav,s2_wav_format,s2_wav_opts = line.split(',')
            separated = utils._process(mix_wav, model)
            s1_hat = separated[:, :, 0][0]
            s2_hat = separated[:, :, 1][0]
            s1 = utils.load_wav(s1_wav)
            s2 = utils.load_wav(s2_wav)
            
            sdr1 = sdr(s1,s1_hat) + sdr(s2,s2_hat)
            sdr2 = sdr(s2,s1_hat) + sdr(s1,s2_hat)
            sdr_tar = sdr1
            if sdr_tar > sdr2:
                sdr_tar = sdr2

            sum_sdr += sdr_tar
            num_sdr += 2

    return sum_sdr / float(num_sdr)

def count_si_sdr(input_csv, model):
    model.modules.eval()
    k=0
    num_sdr = 0.0
    sum_sdr = 0.0
    with open(input_csv) as fp:
        for line in tqdm(fp):
            if k == 0: 
                k += 1
                continue
            line = line.strip()
            ID,duration,mix_wav,mix_wav_format,mix_wav_opts,s1_wav,s1_wav_format,s1_wav_opts,s2_wav,s2_wav_format,s2_wav_opts = line.split(',')
            separated = utils._process(mix_wav, model)
            s1 = utils.load_wav(s1_wav)
            s2 = utils.load_wav(s2_wav)
            target_tensor = torch.stack((s1, s2), dim=1)
            target_tensor = target_tensor.unsqueeze(0)

            si_sdr = get_si_snr_with_pitwrapper(target_tensor, separated)
            
            sum_sdr += si_sdr 
            num_sdr += 1

    return sum_sdr / float(num_sdr)
    pass
if __name__ == '__main__':
    data_type = ['vd']

    model, hparams = utils._load_model()
    data_csv = {'tr': hparams['train_data'] ,\
                'vd': hparams['valid_data'] }

    for dttype in data_type:
        print(dttype)
        print(count_si_sdr(data_csv[dttype], model))
