# Tacotron2

Version of [Tacotron2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2) designed to work with [gruut](https://github.com/rhasspy/gruut).

## Additional Features

* Models can be exported to [onnx](https://onnx.ai/) format
* Inference can output numpy or JSON mels
* Input is strictly phoneme indexes (any text front-end), output is strictly mels (any vocoder)
