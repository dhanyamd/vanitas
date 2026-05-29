import os
from huggingface_hub import snapshot_download
import soundfile as sf
import numpy as np

def test_load():
    repo_path = snapshot_download(repo_id='kyutai/DailyTalkContiguous')
    # Path to first wav file
    wav_path = os.path.join(repo_path, 'data_stereo', '0.wav')
    audio, sr = sf.read(wav_path)
    print('Loaded audio shape:', audio.shape, 'Sample rate:', sr)
    # Verify stereo shape
    if audio.ndim == 2:
        print('Stereo channels:', audio.shape[1])
    else:
        print('Unexpected audio dimensions')

if __name__ == '__main__':
    test_load()
