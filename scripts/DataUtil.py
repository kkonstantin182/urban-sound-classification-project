import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import os
import shutil

SAMPLE_RATE = 44100
#DURATION = 4
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 1024
POWER = 1
LENGTH = 128
INPUT_PATH = "/content/drive/MyDrive/datasets/urban_sound"
OUTPUT_PATH = "/content/drive/MyDrive/UNIMI/MachineLearning/UrbanSound/features/"
OUTPUT_FILE = "spec_feat"


def load_wav(file_path):

    """
    Load waveform from a file.
    """
    waveform, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, res_type="kaiser_fast")
    # cast to fix length
    #waveform = librosa.util.fix_length(waveform, SAMPLE_RATE*DURATION)
    return waveform, sample_rate
    
def get_spectrograms(waveform, sample_rate=SAMPLE_RATE):

    """
    Get spectrograms from a waveform.
    """
    s = librosa.feature.melspectrogram(waveform, sr=sample_rate, n_mels=N_MELS)
    # Convert to log scale
    #log_s = librosa.power_to_db(s, ref=np.max)
    return s

def set_length(array, length = LENGTH):

    """
    Cast an array to a given length.
    """
    return librosa.util.fix_length(array, length)
    
    
def standardize_spectrograms(spectrogram):
    mean = spectrogram.mean()
    std = spectrogram.std()
    s_norm = (spectrogram - mean) / std 
    return s_norm
    
def save_spectrograms(dataframe, 
                     input_path=INPUT_PATH,
                     output_path=OUTPUT_PATH,
                     output_file=OUTPUT_FILE,
                     standartization=False,
                     fix_length=False):
    """
    Apply get_specrtograms function for the files specified in the dataframe.
    :param dataframe: pandas.core.frame.DataFrame
    """
    res = list()
    for index_num, row in tqdm(dataframe.iterrows()):
      file = os.path.join(input_path, "fold"+str(row["fold"]), str(row["slice_file_name"]))
      waveform, sample_rate = load_wav(file)
      feature = get_spectrograms(waveform, sample_rate)
      if standartization == True:
          feature = standardize_spectrograms(feature)
          stand = "with_stand"
      else:
          stand = "without_stand"
      if fix_length == True:
          feature = set_length(feature)
          fl = str(LENGTH)
      else:
          fl = "T"
      label = str(row["class"])
      classID = str(row["classID"])
      dir = str(row["fold"])
      res.append([feature, label, classID, dir])
    
    # file name
    output_file = output_file + f"_{stand}" + f"_{N_MELS}" + f"_{fl}" + f"_{SAMPLE_RATE}" + f"_{N_FFT}" + f"_{HOP_LENGTH}"
    # create a df  
    new_df = pd.DataFrame(res,columns=["feature", "class", "classID", "fold"])
    # convert df to csv
    new_df.to_csv(f"{output_file}.csv")
    # convert df to numpy array
    npar = np.array(new_df)
    np.save(f"{output_file}.npy", npar)
    # save files in Colab
    shutil.copyfile(f"{output_file}.csv", f"{output_path}{output_file}.csv")
    shutil.copyfile(f"{output_file}.npy", f"{output_path}{output_file}.npy")
    return new_df