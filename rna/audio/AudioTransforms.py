# Instalar los siguientes paquetes para poder correr ejecutar las funciones de este módulo

#conda install librosa
#conda install ffmpeg
#conda install audioread

import numpy as np
import librosa as lb


def recortar_silencios(audio):
    # recorta  "silencios" antes y despues del
    # nota: si comparamos dos audios, los silencios dan coincidencias que no deben tomarse en cuenta
    audio_trim, index = lb.effects.trim(audio, top_db=15, frame_length=256, hop_length=128)
    
    return audio_trim



def ajustar_largo_audio(audio, muestreo, tiempo_segs): 
     
    coef_ajuste = (len(audio)/(tiempo_segs*muestreo))
    
    audio_stretch= lb.effects.time_stretch(y=audio, rate=coef_ajuste)
    
    return audio_stretch



def audio_a_espectrograma_mel(audio, muestreo, n_mels):
    hop_length = 256 # Ver
    # use log-melspectrogram
    mels = lb.feature.melspectrogram(y=audio, sr=muestreo, n_mels=n_mels,
                                            n_fft=hop_length*2, hop_length=hop_length)
    #calcula logaritmo y agrega pequeÃ±o valor para evitar valores en 0
    mels = 10*np.log(mels + 1e-9)

    # normalizacion min-max 
    mels = (mels-mels.min()) / (mels.max()-mels.min())
    
    return mels



def espectrograma_mel_a_imagen(mels):
    # asume que mfcc esta en el rango 
    # multiplica por 255 para que quede en el rango de 8-bit
    img = ( 255.0*mels).astype(np.uint8)
    
    # invierte para que las frecuencias bajas queden en la parte inferior
    img = np.flip(img, axis=0) 
    
    # invierte la imagen (mas negro, mas energía)
    img = 255-img
    
    return img