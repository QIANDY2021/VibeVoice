import argparse
import os
import sys
from pathlib import Path
import torch
from math import ceil

sys.path.append(str(Path(__file__).resolve().parents[1]))
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor


def build_voice_tokens(tokenizer, wav_len, compress_ratio):
    vae_tok_len = ceil(wav_len / compress_ratio)
    tokens = []
    tokens += tokenizer.encode(' Voice input:\n', add_special_tokens=False)
    tokens += tokenizer.encode(' Speaker 0:', add_special_tokens=False)
    tokens += [tokenizer.speech_start_id]
    tokens += [tokenizer.speech_diffusion_id] * vae_tok_len
    tokens += [tokenizer.speech_end_id]
    tokens += tokenizer.encode('\n', add_special_tokens=False)
    tokens += tokenizer.encode(' Speech output:\n', add_special_tokens=False)
    tokens += [tokenizer.speech_start_id]
    return tokens


def generate_prefilled(model, tokenizer, input_ids):
    device = next(model.parameters()).device
    ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.ones_like(ids, device=device)
    lm_out = model.forward_lm(input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True)
    tts_mask = torch.ones_like(ids[:, -1:], device=device)
    tts_out = model.forward_tts_lm(input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True, tts_text_masks=tts_mask, lm_last_hidden_state=lm_out.last_hidden_state)
    return lm_out, tts_out


def generate_negative_prefilled(model, tokenizer):
    device = next(model.parameters()).device
    neg_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    ids = torch.tensor([[neg_id]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, device=device)
    lm_out = model.forward_lm(input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True)
    tts_mask = torch.ones_like(ids[:, -1:], device=device)
    tts_out = model.forward_tts_lm(input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True, tts_text_masks=tts_mask, lm_last_hidden_state=lm_out.last_hidden_state)
    return lm_out, tts_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='microsoft/VibeVoice-Realtime-0.5B')
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--speaker_key', type=str, required=True)
    parser.add_argument('--device', type=str, default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
    parser.add_argument('--inference_steps', type=int, default=5)
    args = parser.parse_args()

    if args.device.lower() == 'mpx':
        args.device = 'mps'
    if args.device == 'mps' and not torch.backends.mps.is_available():
        args.device = 'cpu'

    processor = VibeVoiceStreamingProcessor.from_pretrained(args.model_path)

    if args.device == 'mps':
        load_dtype = torch.float32
        device_map = None
        attn_impl = 'sdpa'
    elif args.device == 'cuda':
        load_dtype = torch.bfloat16
        device_map = 'cuda'
        attn_impl = 'flash_attention_2'
    else:
        load_dtype = torch.float32
        device_map = 'cpu'
        attn_impl = 'sdpa'

    try:
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            args.model_path,
            torch_dtype=load_dtype,
            device_map=device_map,
            attn_implementation=attn_impl,
        )
        if args.device == 'mps':
            model.to('mps')
    except Exception:
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            args.model_path,
            torch_dtype=load_dtype,
            device_map=args.device,
            attn_implementation='sdpa',
        )

    model.eval()
    model.set_ddpm_inference_steps(num_steps=args.inference_steps)

    audio_cfg = processor.audio_processor
    wav = audio_cfg._load_audio_from_path(args.audio_path)
    if processor.db_normalize and processor.audio_normalizer:
        wav = processor.audio_normalizer(wav)

    tokens = build_voice_tokens(processor.tokenizer, wav.shape[0], processor.speech_tok_compress_ratio)

    lm_out, tts_out = generate_prefilled(model, processor.tokenizer, tokens)
    neg_lm_out, neg_tts_out = generate_negative_prefilled(model, processor.tokenizer)

    save_dir = os.path.join(os.path.dirname(__file__), 'voices', 'streaming_model')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{args.speaker_key}.pt')
    torch.save({
        'lm': lm_out,
        'tts_lm': tts_out,
        'neg_lm': neg_lm_out,
        'neg_tts_lm': neg_tts_out,
    }, save_path)
    print(save_path)


if __name__ == '__main__':
    main()