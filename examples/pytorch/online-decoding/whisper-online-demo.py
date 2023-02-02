import argparse
import torch
import torchaudio
from transformers.generation_utils import EvaluatedHypotheses
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def longest_common_prefix(o,t, eos):
        pref = []
        for p, d in zip(o, t):
            if p == d and p != eos:
                pref.append(p)
            else:
                break
        return pref

def main(args):
    # load model and processor
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.to(args.device)
    model.config.forced_decoder_ids = None

    samples, sr = torchaudio.load(args.audio_path)

    input_features = processor(samples[0], sampling_rate=sr, return_tensors="pt").input_features.to(args.device) 
    predicted_ids = model.generate(input_features)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    print('Offline:', transcription)
    eos = predicted_ids[0,-1]

    eh = EvaluatedHypotheses(repetition_detection=args.repetition_detection)
    unstable = predicted_ids[0,:4] # check out the Whisper's documentation
    stable = predicted_ids[0,:4]

    STEP_SIZE = int(sr * args.chunk_size)

    idx = 0
    while idx < samples.shape[-1]:
        idx += STEP_SIZE
        input_features = processor(samples[0,:idx], sampling_rate=sr, return_tensors="pt").input_features
        input_features = input_features.to(args.device)
        decoder_input_ids = stable.unsqueeze(0).to(args.device)
        if idx >= samples.shape[-1]:
            eh = None
        out = model.generate(input_features, decoder_input_ids=decoder_input_ids, evaluated_hypotheses=eh, num_return_sequences=1, num_beams=6)
        if eh: eh.finish_step()
        h = out[0].cpu()

        stable = h
        #do LCP if not finished
        if idx < samples.shape[1]:
            stable = torch.LongTensor(longest_common_prefix(unstable, h[:-2], eos)) # -2 remove EOS and one more token before
        print(f'Input {idx/sr}s out of {samples.shape[-1]/sr}s')
        print('Hypo:  ', processor.decode(h, skip_special_tokens=True))
        print('Stable:', processor.decode(stable, skip_special_tokens=True))
        print('*'*50,'\n')
        unstable = h

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path', type=str)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--chunk-size', type=float, default=0.3, help='Chunk size in seconds')
    parser.add_argument('--model', type=str, default='openai/whisper-base')
    parser.add_argument('--repetition-detection', action='store_true')
    main(parser.parse_args())