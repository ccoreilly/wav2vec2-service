import torch, transformers, ctc_segmentation
import soundfile
import json

model_file = 'ccoreilly/wav2vec2-large-100k-voxpopuli-catala'
vocab_dict = None
with open('./vocab.json') as f:
    vocab_dict = json.load(f)
processor = transformers.Wav2Vec2Processor.from_pretrained(model_file)
model = transformers.Wav2Vec2ForCTC.from_pretrained(model_file)

wav = "bon-vespra.wav"
speech_array, sampling_rate = soundfile.read( wav )
assert sampling_rate == 16000
# Generate a transcription, if not yet available
# (Note that this will introduce errors if the model is wrong)
features = processor(speech_array, sampling_rate=16000, return_tensors="pt")
input_values = features.input_values
attention_mask = features.attention_mask
with torch.no_grad():
    logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]
print(f"Transcription: {transcription}")
# Split the transcription into words and handle as separate utterances
text = transcription.split()
# CTC log posteriors inference
with torch.no_grad():
    softmax = torch.nn.LogSoftmax(dim = -1)
    lpz = softmax(logits)[0].cpu().numpy()
print(lpz)
# CTC segmentation preparation
char_list = [x.lower() for x in vocab_dict.keys()]
print(char_list)
config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
config.index_duration = speech_array.shape[0] / lpz.shape[0] / sampling_rate
print(f"Duration {config.index_duration * lpz.shape[0]}")
# CTC segmentation
ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, text)
print(ground_truth_mat)
print(utt_begin_indices)
timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, lpz, ground_truth_mat)
segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text)
# print segments
for word, segment in zip(text, segments):
    print(f"{segment[0]:.2f} {segment[1]:.2f} {segment[2]:3.4f} {word}")