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

import click
from pyvisqol import Visqol

from tts_pqe.speech_recognition import SpeechRecognition
from tts_pqe.speaker_identification import SpeakerIdentification


@click.command()
@click.argument("reference", type=click.Path(exists=True, file_okay=True))
@click.argument("defraded", type=click.Path(exists=True, file_okay=True))
def main(reference, defraded):
    moslqo = Visqol().measure(reference, defraded)
    similarity = SpeakerIdentification().similarity(reference, defraded)
    text1, text2, editdistance = SpeechRecognition().wer(reference, defraded)
    print("MOS:", moslqo)
    print("Similarity:", similarity)
    print("Edit distance", editdistance)
    print(f"ASR result of {reference}:\n", text1)
    print(f"ASR result of {defraded}:\n", text2)


if __name__ == "__main__":
    main()
