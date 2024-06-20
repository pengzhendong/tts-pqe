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

import numpy as np
import sherpa_onnx
import soundfile as sf
from modelscope.hub.file_download import model_file_download


class SpeakerIdentification:
    def __init__(self):
        model = model_file_download(
            "pengzhendong/speaker-identification", "wespeaker_zh_cnceleb_resnet34.onnx"
        )
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model, num_threads=1, provider="cpu"
        )
        if not config.validate():
            raise ValueError(f"Invalid config. {config}")
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def compute(self, wav_path):
        audio, sample_rate = sf.read(wav_path, dtype=np.float32)
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=audio)
        stream.input_finished()
        assert self.extractor.is_ready(stream)
        return np.array(self.extractor.compute(stream))

    def similarity(self, reference, defraded):
        embedding1 = self.compute(reference)
        embedding2 = self.compute(defraded)
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
