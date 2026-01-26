import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import soundfile

TOKEN = ''


def get_diarization(output_audio):
    # output_audio = './output_audio/Рафаэль_Рише_покинул_＂Трактор＂_День_с_Алексеем_Шевченко_Bytpl2vSpNE.wav'
    print('до загрузки модели')
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-community-1', token=TOKEN)
    print('после загрузки модели')
    device = torch.device('cpu')
    print('после создания ус-ва')
    pipeline.to(device)
    print('после привязки ус-ва')
    #output = pipeline(output_audio, num_speakers=2, exclusive=True)
    wave_form_np, sample_rate = soundfile.read(output_audio, dtype='float32')
    print('считали аудио файл')
    wave_form = torch.from_numpy(wave_form_np).unsqueeze(0)
    print('преобразовали аудио данные')
    audio_dict = {'waveform': wave_form, 'sample_rate': sample_rate}
    print(wave_form.shape)
    with ProgressHook() as hook:
        output = pipeline(audio_dict, hook=hook)
    print('выполнили диаризацию')
    segments = []

    for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
        segments.append({'speaker': str(speaker), 'start': float(turn.start), 'end': float(turn.end)})

    #print(segments)
    speakers = {}
    for segment in segments:
        if segment['speaker'] not in speakers:
            speakers[segment['speaker']] = {'segments': []}
        speakers[segment['speaker']]['segments'].append((segment['start'], segment['end']))

    return speakers

# print (get_diarization('./output_audio/Рафаэль_Рише_покинул_＂Трактор＂_День_с_Алексеем_Шевченко_Bytpl2vSpNE.wav'))

#
# [{'speaker': 'SPEAKER_02', 'start': 0.03096875, 'end': 3.42284375},
#  {'speaker': 'SPEAKER_00', 'start': 4.199093749999999, 'end': 8.586593750000002},
#  {'speaker': 'SPEAKER_00', 'start': 8.89034375, 'end': 13.193468750000001},
#  {'speaker': 'SPEAKER_02', 'start': 18.96471875, 'end': 19.977218750000002},
#  {'speaker': 'SPEAKER_03', 'start': 23.38596875, 'end': 23.402843750000002},
#  {'speaker': 'SPEAKER_02', 'start': 23.402843750000002, 'end': 23.85846875},
#  {'speaker': 'SPEAKER_01', 'start': 32.751593750000005, 'end': 42.16784375},
#  {'speaker': 'SPEAKER_01', 'start': 45.25596875, 'end': 57.861593750000004},
#  {'speaker': 'SPEAKER_01', 'start': 58.485968750000005, 'end': 163.66784375},
#  {'speaker': 'SPEAKER_01', 'start': 164.12346875, 'end': 175.56471875},
#  {'speaker': 'SPEAKER_01', 'start': 178.85534375, 'end': 311.89784375},
#  {'speaker': 'SPEAKER_01', 'start': 315.12096875000003, 'end': 328.67159375},
#  {'speaker': 'SPEAKER_03', 'start': 337.12596875, 'end': 349.09034375000005},
#  {'speaker': 'SPEAKER_03', 'start': 349.47846875000005, 'end': 352.33034375},
#  {'speaker': 'SPEAKER_02', 'start': 351.87471875, 'end': 352.49909375000004},
#  {'speaker': 'SPEAKER_03', 'start': 352.49909375000004, 'end': 352.65096875},
#  {'speaker': 'SPEAKER_02', 'start': 352.65096875, 'end': 352.73534375},
#  {'speaker': 'SPEAKER_02', 'start': 353.27534375000005, 'end': 361.91534375000003},
#  {'speaker': 'SPEAKER_03', 'start': 358.67534375, 'end': 358.87784375},
#  {'speaker': 'SPEAKER_03', 'start': 358.92846875000004, 'end': 358.96221875000003},
#  {'speaker': 'SPEAKER_03', 'start': 359.06346875, 'end': 359.78909375},
#  {'speaker': 'SPEAKER_02', 'start': 362.59034375000005, 'end': 365.13846875},
#  {'speaker': 'SPEAKER_03', 'start': 365.42534375, 'end': 370.96034375000005}]
