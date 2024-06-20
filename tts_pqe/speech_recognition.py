# Copyright (c) 2024, Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import editdistance
import librosa
import sherpa_onnx
from modelscope import snapshot_download


class SpeechRecognition:
    def __init__(self):
        self.sample_rate = 16000
        asr_repo_dir = snapshot_download("pengzhendong/offline-paraformer-zh")
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=f"{asr_repo_dir}/model.onnx",
            tokens=f"{asr_repo_dir}/tokens.txt",
            num_threads=1,
            sample_rate=self.sample_rate,
            feature_dim=80
        )

    def compute(self, wav_path):
        audio, _ = librosa.load(wav_path, sr=self.sample_rate)
        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, audio)
        self.recognizer.decode_stream(stream)
        return stream.result.text

    def wer(self, reference, defraded):
        text1 = self.compute(reference)
        text2 = self.compute(defraded)
        return text1, text2, editdistance.eval(text1, text2)
