import os
import ffmpeg
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from diarization import get_diarization
from voice_assessment import assess_voices
from qwen import describe_all_voices_with_qwen
import json

# export DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix ffmpeg)/lib"
video_path = './input_video'
audio_path = './output_audio'
data_path = './output_data'

def get_data():
    os.makedirs(audio_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True
    )

    for item in os.listdir(video_path):
        video_file = os.path.join(video_path, item)
        file_name = item.split('.')[0]
        if os.path.isfile(video_file):
            output_audio = os.path.join(audio_path, f'{file_name}.wav')
            ffmpeg.input(video_file).output(output_audio, vn=None, acodec='pcm_s16le', ar=16000, ac=1).overwrite_output().run()
            result = pipe(output_audio)
            #print(result["text"])
            # with open(f'{file_name}.txt', 'w') as f:
            #     f.write(result["text"])
            diarization_data = get_diarization(output_audio)

            data = {
                'transcription': result['text'],
                'diarization': diarization_data
            }

            # --- Voice assessment + LLM description (per speaker) ---
            metrics_by_speaker = assess_voices(data, audio_path=output_audio)
            for speaker, metrics in metrics_by_speaker.items():
                if speaker in data["diarization"] and isinstance(data["diarization"][speaker], dict):
                    data["diarization"][speaker]["voice_metrics"] = metrics

            model_path = os.environ.get("QWEN_MODEL_PATH")
            if model_path:
                descriptions = describe_all_voices_with_qwen(metrics_by_speaker, model_path=model_path)
                for speaker, desc in descriptions.items():
                    if speaker in data["diarization"] and isinstance(data["diarization"][speaker], dict):
                        data["diarization"][speaker]["voice_description"] = desc
            else:
                # If model isn't configured, keep the field present but empty.
                for speaker in data["diarization"].keys():
                    if isinstance(data["diarization"][speaker], dict):
                        data["diarization"][speaker]["voice_description"] = ""
            output_data = os.path.join(data_path, f'{file_name}.json')
            with open(output_data, 'w') as file:
                file.write(json.dumps(data, indent=4))


if __name__ == "__main__":
    get_data()