#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: furushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import angles
from contextlib import contextmanager
from typing import List
import usb.core
import usb.util
import pyaudio
import math
import numpy as np
import rclpy.time
import tf_transformations as T
import os
import rclpy
from rclpy.clock import ClockType
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
from rcl_interfaces.msg import SetParametersResult
from std_srvs.srv import SetBool
import struct
import sys
import time
from audio_common_msgs.msg import AudioData
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Int32, ColorRGBA
from pepper_interfaces.msg import MetaMic
from ros_utils import *
from pepper_nodes.respeaker.config_dict import RESPEAKER_PARAMETERS_ROS

# importa libreria per controllare i led
try:
    from pixel_ring import usb_pixel_ring_v2
except IOError as e:
    print(e)
    raise RuntimeError("Check the device is connected and recognized")


# suppress error messages from ALSA
# https://stackoverflow.com/questions/7088672/pyaudio-working-but-spits-out-error-messages-each-time
# https://stackoverflow.com/questions/36956083/how-can-the-terminal-output-of-executables-run-by-python-functions-be-silenced-i
@contextmanager
def ignore_stderr(enable=True):
    if enable:
        devnull = None
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            stderr = os.dup(2)
            sys.stderr.flush()
            os.dup2(devnull, 2)
            try:
                yield
            finally:
                os.dup2(stderr, 2)
                os.close(stderr)
        finally:
            if devnull is not None:
                os.close(devnull)
    else:
        yield


# Partly copied from https://github.com/respeaker/usb_4_mic_array
# parameter list
# name: (id, offset, type, max, min , r/w, info)
# dizionario per controllare i chip del microfono
RESPEAKER_PARAMETERS = {
    'AECFREEZEONOFF': (18, 7, 'int', 1, 0, 'rw', 'Adaptive Echo Canceler updates inhibit.', '0 = Adaptation enabled', '1 = Freeze adaptation, filter only'),
    'AECNORM': (18, 19, 'float', 16, 0.25, 'rw', 'Limit on norm of AEC filter coefficients'),
    'AECPATHCHANGE': (18, 25, 'int', 1, 0, 'ro', 'AEC Path Change Detection.', '0 = false (no path change detected)', '1 = true (path change detected)'),
    'RT60': (18, 26, 'float', 0.9, 0.25, 'ro', 'Current RT60 estimate in seconds'),
    'HPFONOFF': (18, 27, 'int', 3, 0, 'rw', 'High-pass Filter on microphone signals.', '0 = OFF', '1 = ON - 70 Hz cut-off', '2 = ON - 125 Hz cut-off', '3 = ON - 180 Hz cut-off'),
    'RT60ONOFF': (18, 28, 'int', 1, 0, 'rw', 'RT60 Estimation for AES. 0 = OFF 1 = ON'),
    'AECSILENCELEVEL': (18, 30, 'float', 1, 1e-09, 'rw', 'Threshold for signal detection in AEC [-inf .. 0] dBov (Default: -80dBov = 10log10(1x10-8))'),
    'AECSILENCEMODE': (18, 31, 'int', 1, 0, 'ro', 'AEC far-end silence detection status. ', '0 = false (signal detected) ', '1 = true (silence detected)'),
    'AGCONOFF': (19, 0, 'int', 1, 0, 'rw', 'Automatic Gain Control. ', '0 = OFF ', '1 = ON'),
    'AGCMAXGAIN': (19, 1, 'float', 1000, 1, 'rw', 'Maximum AGC gain factor. ', '[0 .. 60] dB (default 30dB = 20log10(31.6))'),
    'AGCDESIREDLEVEL': (19, 2, 'float', 0.99, 1e-08, 'rw', 'Target power level of the output signal. ', '[-inf .. 0] dBov (default: -23dBov = 10log10(0.005))'),
    'AGCGAIN': (19, 3, 'float', 1000, 1, 'rw', 'Current AGC gain factor. ', '[0 .. 60] dB (default: 0.0dB = 20log10(1.0))'),
    'AGCTIME': (19, 4, 'float', 1, 0.1, 'rw', 'Ramps-up / down time-constant in seconds.'),
    'CNIONOFF': (19, 5, 'int', 1, 0, 'rw', 'Comfort Noise Insertion.', '0 = OFF', '1 = ON'),
    'FREEZEONOFF': (19, 6, 'int', 1, 0, 'rw', 'Adaptive beamformer updates.', '0 = Adaptation enabled', '1 = Freeze adaptation, filter only'),
    'STATNOISEONOFF': (19, 8, 'int', 1, 0, 'rw', 'Stationary noise suppression.', '0 = OFF', '1 = ON'),
    'GAMMA_NS': (19, 9, 'float', 3, 0, 'rw', 'Over-subtraction factor of stationary noise. min .. max attenuation'),
    'MIN_NS': (19, 10, 'float', 1, 0, 'rw', 'Gain-floor for stationary noise suppression.', '[-inf .. 0] dB (default: -16dB = 20log10(0.15))'),
    'NONSTATNOISEONOFF': (19, 11, 'int', 1, 0, 'rw', 'Non-stationary noise suppression.', '0 = OFF', '1 = ON'),
    'GAMMA_NN': (19, 12, 'float', 3, 0, 'rw', 'Over-subtraction factor of non- stationary noise. min .. max attenuation'),
    'MIN_NN': (19, 13, 'float', 1, 0, 'rw', 'Gain-floor for non-stationary noise suppression.', '[-inf .. 0] dB (default: -10dB = 20log10(0.3))'),
    'ECHOONOFF': (19, 14, 'int', 1, 0, 'rw', 'Echo suppression.', '0 = OFF', '1 = ON'),
    'GAMMA_E': (19, 15, 'float', 3, 0, 'rw', 'Over-subtraction factor of echo (direct and early components). min .. max attenuation'),
    'GAMMA_ETAIL': (19, 16, 'float', 3, 0, 'rw', 'Over-subtraction factor of echo (tail components). min .. max attenuation'),
    'GAMMA_ENL': (19, 17, 'float', 5, 0, 'rw', 'Over-subtraction factor of non-linear echo. min .. max attenuation'),
    'NLATTENONOFF': (19, 18, 'int', 1, 0, 'rw', 'Non-Linear echo attenuation.', '0 = OFF', '1 = ON'),
    'NLAEC_MODE': (19, 20, 'int', 2, 0, 'rw', 'Non-Linear AEC training mode.', '0 = OFF', '1 = ON - phase 1', '2 = ON - phase 2'),
    'SPEECHDETECTED': (19, 22, 'int', 1, 0, 'ro', 'Speech detection status.', '0 = false (no speech detected)', '1 = true (speech detected)'),
    'FSBUPDATED': (19, 23, 'int', 1, 0, 'ro', 'FSB Update Decision.', '0 = false (FSB was not updated)', '1 = true (FSB was updated)'),
    'FSBPATHCHANGE': (19, 24, 'int', 1, 0, 'ro', 'FSB Path Change Detection.', '0 = false (no path change detected)', '1 = true (path change detected)'),
    'TRANSIENTONOFF': (19, 29, 'int', 1, 0, 'rw', 'Transient echo suppression.', '0 = OFF', '1 = ON'),
    'VOICEACTIVITY': (19, 32, 'int', 1, 0, 'ro', 'VAD voice activity status.', '0 = false (no voice activity)', '1 = true (voice activity)'),
    'STATNOISEONOFF_SR': (19, 33, 'int', 1, 0, 'rw', 'Stationary noise suppression for ASR.', '0 = OFF', '1 = ON'),
    'NONSTATNOISEONOFF_SR': (19, 34, 'int', 1, 0, 'rw', 'Non-stationary noise suppression for ASR.', '0 = OFF', '1 = ON'),
    'GAMMA_NS_SR': (19, 35, 'float', 3, 0, 'rw', 'Over-subtraction factor of stationary noise for ASR. ', '[0.0 .. 3.0] (default: 1.0)'),
    'GAMMA_NN_SR': (19, 36, 'float', 3, 0, 'rw', 'Over-subtraction factor of non-stationary noise for ASR. ', '[0.0 .. 3.0] (default: 1.1)'),
    'MIN_NS_SR': (19, 37, 'float', 1, 0, 'rw', 'Gain-floor for stationary noise suppression for ASR.', '[-inf .. 0] dB (default: -16dB = 20log10(0.15))'),
    'MIN_NN_SR': (19, 38, 'float', 1, 0, 'rw', 'Gain-floor for non-stationary noise suppression for ASR.', '[-inf .. 0] dB (default: -10dB = 20log10(0.3))'),
    'GAMMAVAD_SR': (19, 39, 'float', 1000, 0, 'rw', 'Set the threshold for voice activity detection.', '[-inf .. 60] dB (default: 3.5dB 20log10(1.5))'),
    # 'KEYWORDDETECT': (20, 0, 'int', 1, 0, 'ro', 'Keyword detected. Current value so needs polling.'),
    'DOAANGLE': (21, 0, 'int', 359, 0, 'ro', 'DOA angle. Current value. Orientation depends on build configuration.')
}


#configurazione microfono e lettura VAD DOA
#va in esecuzione una sola volta all'avvio, ad eccezione dei metodi is_voice e direction
class RespeakerInterface(object):
    VENDOR_ID = 0x2886
    PRODUCT_ID = 0x0018
    TIMEOUT = 100000 # 100 secondi, se l'USB non risponde entro questo tempo dà errore

    def __init__(self, node: Node):
        self.node = node

        # Cerca tra tutte le periferiche USB connesse quella con quel Vendor e Product ID.
        self.dev = usb.core.find(idVendor=self.VENDOR_ID,
                                 idProduct=self.PRODUCT_ID)
        if not self.dev:
            raise RuntimeError("Failed to find Respeaker device")
        self.node.get_logger().info("Initializing Respeaker device")

        # Esegue un reset hardware della porta USB per pulire stati "appesi" precedenti.
        self.dev.reset()

        # Inizializza la libreria per controllare i LED colorati (l'anello di luci).
        self.pixel_ring = usb_pixel_ring_v2.PixelRing(self.dev)
        self.set_led_think()

        #Utile dopo il reset, blocca il codice del nodo.
        time.sleep(2)  # it will take 10 seconds to re-recognize as audio device

        # Accende i LED in modalità "Trace" (ascolto).
        self.set_led_trace()
        self.node.get_logger().info("Respeaker device initialized (Version: %s)" % self.version)

    #invocato quando clicchi ctrl + c
    def __del__(self):
        try:
            self.close()
        except:
            pass
        finally:
            self.dev = None

    def write(self, name, value):
        try:
            data = RESPEAKER_PARAMETERS[name]
        except KeyError:
            return

        if data[5] == 'ro':
            raise ValueError('{} is read-only'.format(name))

        id = data[0]

        if data[2] == 'int':
            payload = struct.pack(b'iii', data[1], int(value), 1)
        else:
            payload = struct.pack(b'ifi', data[1], float(value), 0)

        # --- INIZIO MODIFICA WRITE ---
        MAX_RETRIES = 5
        for attempt in range(MAX_RETRIES):
            try:
                self.dev.ctrl_transfer(
                    usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                    0, 0, id, payload, self.TIMEOUT)
                break
            except usb.core.USBError as e:
                if attempt < MAX_RETRIES - 1:
                    self.node.get_logger().warn(f"USB Write timeout su {name}, riprovo... ({attempt+1}/{MAX_RETRIES})")
                    time.sleep(0.05)
                else:
                    self.node.get_logger().fatal("Mic USB Write error irreversibile.")
                    self.node.destroy_node()
                    raise usb.core.USBError(e.strerror)
        # --- FINE MODIFICA ---

    #Serve per chiedere al chip lo stato attuale (es. "C'è voce?", "Qual è l'angolo?")
    def read(self, name):
        try:
            data = RESPEAKER_PARAMETERS[name]
        except KeyError:
            return

        id = data[0]

        cmd = 0x80 | data[1]
        if data[2] == 'int':
            cmd |= 0x40

        length = 8

        # --- INIZIO MODIFICA ANTI-CRASH ---
        response = None
        MAX_RETRIES = 5  # Numero di tentativi prima di arrendersi

        for attempt in range(MAX_RETRIES):
            try:
                # Tenta la lettura USB
                response = self.dev.ctrl_transfer(
                    usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                    0, cmd, id, length, self.TIMEOUT)
                
                # Se siamo arrivati qui, ha funzionato! Usciamo dal ciclo.
                break 

            except usb.core.USBError as e:
                # Se siamo qui, c'è stato un errore (es. Timeout)
                if attempt < MAX_RETRIES - 1:
                    # Se non è l'ultimo tentativo, logga un warning e aspetta
                    self.node.get_logger().warn(f"USB Read timeout su {name}, riprovo... ({attempt+1}/{MAX_RETRIES})")
                    time.sleep(0.05) # Aspetta 50 millisecondi (impercettibile per l'utente)
                else:
                    # Se siamo all'ultimo tentativo e fallisce ancora, ALLORA chiudi tutto.
                    self.node.get_logger().fatal(f"Mic USB error irreversibile dopo {MAX_RETRIES} tentativi.")
                    self.node.destroy_node()
                    raise usb.core.USBError(e.strerror)
        # --- FINE MODIFICA ---

        response = struct.unpack(b'ii', response.tobytes())

        if data[2] == 'int':
            result = response[0]
        else:
            result = response[0] * (2.**response[1])

        return result

    #chiamate per i led
    def set_led_think(self):
        self.pixel_ring.set_brightness(10)
        self.pixel_ring.think()

    def set_led_trace(self):
        self.pixel_ring.set_brightness(20)
        self.pixel_ring.trace()

    def set_led_color(self, r, g, b, a):
        self.pixel_ring.set_brightness(int(20 * a))
        self.pixel_ring.set_color(r=int(r*255), g=int(g*255), b=int(b*255))

    #imposta soglia voce
    def set_vad_threshold(self, db):
        self.write('GAMMAVAD_SR', db)

    #chiede se c'è voce
    def is_voice(self):
        return self.read('VOICEACTIVITY')

    #chiede la direzione
    @property
    def direction(self):
        return self.read('DOAANGLE')

    @property
    def version(self):
        return self.dev.ctrl_transfer(
            usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, 0x80, 0, 1, self.TIMEOUT)[0]

    def close(self):
        """
        close the interface
        libera risorse USB
        """
        usb.util.dispose_resources(self.dev)

# classe che si occupa di catturare audio grezzi
class RespeakerAudio():
    def __init__(self, node: Node, on_audio, channels=None, suppress_error=True):
        self.node = node
        self.logger = self.node.get_logger()

        # SALVA LA FUNZIONE DI CALLBACK.
        # "on_audio" è una funzione del nodo principale (RespeakerNode).
        # Questa classe la chiamerà ogni volta che ha dei dati audio pronti.
        self.on_audio = on_audio

        # Crea un publisher per dire agli altri nodi le specifiche del microfono (es. 16000Hz)
        self.pub_meta = self.node.create_publisher(MetaMic, "/meta/mic", qos_profile=LATCH_PROFILE)

        # Inizializza PyAudio (la libreria che parla con la scheda audio).
        # "ignore_stderr" nasconde i messaggi di errore inutili di Linux ALSA.
        with ignore_stderr(enable=suppress_error):
            self.pyaudio = pyaudio.PyAudio()

        # Impostazioni standard per la voce    
        self.available_channels = None
        self.channels = channels
        self.device_index = None
        self.rate = 16000
        self.chunk = 1024 # Legge l'audio a blocchetti di 1024 campioni alla volta
        self.bitwidth = 2
        self.bitdepth = 16
        self.publish_meta()

        # --- NUOVA LOGICA DI RICERCA DEBUGGABILE ---
        count = self.pyaudio.get_device_count()
        self.logger.info(f"AUDIO SCAN: Found {count} devices. Searching for ReSpeaker...")
        
        self.device_index = None
        
        for i in range(count):
            info = self.pyaudio.get_device_info_by_index(i)
            name = info["name"]
            chan = info["maxInputChannels"]
            
            # Stampa TUTTO quello che vede per capire il problema
            # Usiamo 'lower()' per ignorare maiuscole/minuscole
            is_respeaker_name = "respeaker" in name.lower()
            
            self.logger.info(f" -> Dev {i}: '{name}' | Chan: {chan} | MatchName: {is_respeaker_name}")

            # STRATEGIA DI SELEZIONE:
            # 1. Deve avere "respeaker" nel nome
            # 2. DEVE avere canali > 0 (altrimenti è l'uscita audio)
            if is_respeaker_name and chan > 0:
                self.available_channels = chan
                self.device_index = i
                self.logger.info(f"✅ SUCCESS: Found ReSpeaker at Index {i} with {chan} channels.")
                break
        
        # FALLBACK INTELLIGENTE:
        # Se non lo trova per nome, prova a cercare un dispositivo con ESATTAMENTE 6 canali (firma tipica del ReSpeaker)
        if self.device_index is None:
            self.logger.warn("Name match failed. Trying to find by channel count (6)...")
            for i in range(count):
                info = self.pyaudio.get_device_info_by_index(i)
                if info["maxInputChannels"] == 6:
                    self.available_channels = 6
                    self.device_index = i
                    self.logger.warn(f"⚠️  Found device by CHANNEL COUNT only (Index {i}: {info['name']}). Using it.")
                    break

        # FALLBACK FINALE (Sistema Default)
        if self.device_index is None:
            self.logger.error("❌ IMPOSSIBLE TO FIND RESPEAKER. Falling back to System Default.")
            info = self.pyaudio.get_default_input_device_info()
            self.available_channels = info["maxInputChannels"]
            self.device_index = info["index"]
            self.logger.warn(f"Using System Default: Index {self.device_index}, Channels {self.available_channels}")

        # --- FINE BLOCCO RICERCA ---

        # Controllo di sicurezza: Il ReSpeaker v2 DEVE avere 6 canali.
        if self.available_channels != 6:
            self.logger.warn("%d channel is found for respeaker" % self.available_channels)
            self.logger.warn("You may have to update firmware.")
        
        # Se non specifichiamo canali, li prendiamo tutti (0,1,2,3,4,5)
        if self.channels is None:
            self.channels = range(self.available_channels)
        else:
            self.channels = filter(lambda c: 0 <= c < self.available_channels, self.channels)
        if not self.channels:
            raise RuntimeError('Invalid channels %s. (Available channels are %s)' % (
                self.channels, self.available_channels))
        self.logger.info('Using channels %s' % self.channels)

        # Qui avviene la magia. Apre la connessione con la scheda audio.
        self.stream = self.pyaudio.open(
            input=True, start=False, # start=False: non inizia subito a registrare
            format=pyaudio.paInt16,
            channels=self.available_channels, # Registra tutti i canali
            rate=self.rate,
            frames_per_buffer=self.chunk, # Dimensione del "sorso" d'audio
            stream_callback=self.stream_callback, # <--- IMPORTANTE: Funzione da chiamare quando i dati sono pronti
            input_device_index=self.device_index,
        )

    def publish_meta(self):
        # Manda un messaggio ROS con Rate e Chunk size.
        msg = MetaMic(
            sample_rate=self.rate,
            chunks_len=self.chunk
        )
        self.pub_meta.publish(msg)
    
    def __del__(self):
        # Viene chiamato quando spegni il nodo.
        self.stop()
        try:
            self.stream.close()
        except:
            pass
        finally:
            self.stream = None
        try:
            self.pyaudio.terminate()
        except:
            pass

    def stream_callback(self, in_data, frame_count, time_info, status):
        # split channel

        # Converte i byte in numeri interi (Numpy Array)
        data = np.frombuffer(in_data, dtype=np.int16)

        # Calcola quanti campioni ci sono per ogni canale
        chunk_per_channel = len(data) // self.available_channels

        # --- SEPARAZIONE DEI CANALI (DE-INTERLEAVING) ---
        # Trasforma la lista piatta in una matrice (Righe x Colonne)
        # Ora abbiamo una colonna per ogni microfono.
        data = np.reshape(data, (chunk_per_channel, self.available_channels))
        for chan in self.channels:
            chan_data = data[:, chan]

            # invoke callback
            # --- PASSAGGIO AL NODO ROS ---
            # Chiama la funzione del nodo principale e gli passa i dati.
            # Questo fa scattare "on_audio" dentro RespeakerNode.
            self.on_audio(chan_data.tobytes(), chan)
        return None, pyaudio.paContinue

    def start(self):
        # Fa partire lo stream.
        if self.stream.is_stopped():
            self.stream.start_stream()

    def stop(self):
        # Mette in pausa lo stream.
        if self.stream.is_active():
            self.stream.stop_stream()


class RespeakerNode(Node):

    # --- DEFINIZIONE PARAMETRI ROS ---
    # Questi sono parametri configurabili dal file .yaml o da riga di comando.
    # Non controllano direttamente il chip, ma la logica del nodo (es. tempi di silenzio).
    ADDITIONAL_ROS_PARAMETERS = ROS2ParamList(
        ROS2Param("update_rate", dest="update_rate", value=10.0),            # Frequenza del ciclo di controllo (10 Hz = 0.1 secondi)
        ROS2Param("sensor_frame_id", dest="sensor_frame_id", value="respeaker_base"), # Nome del frame TF per la localizzazione 3D
        ROS2Param("doa_xy_offset", dest="doa_xy_offset", value=0.0),         # Offset fisico se il mic non è al centro esatto
        ROS2Param("doa_yaw_offset", dest="doa_yaw_offset", value=90.0),      # Rotazione del microfono (spesso montato ruotato di 90 gradi)
        
        # PARAMETRI PER IL "TAGLIO" DELLE FRASI (VAD Logic)
        ROS2Param("speech_prefetch", dest="speech_prefetch", value=3.0),     # Quanti secondi di audio tenere in memoria "prima" che inizi a parlare (per non perdere l'inizio della frase)
        ROS2Param("speech_continuation", dest="speech_continuation", value=2.0), # Quanti secondi di silenzio aspettare prima di dire "Ok, ha finito di parlare"
        ROS2Param("max_duration", dest="speech_max_duration", value=60),     # Durata massima di una registrazione
        ROS2Param("min_duration", dest="speech_min_duration", value=0.1),    # Durata minima (ignora rumori brevissimi tipo colpi di tosse)
        ROS2Param("main_channel", dest="main_channel", value=0),             # Quale canale usare per il riconoscimento vocale (0 è solitamente quello processato)
        ROS2Param("suppress_pyaudio_error", dest="suppress_pyaudio_error", value=True), # Nasconde errori brutti nel terminale
    )

    # Unisce i parametri hardware (dalla classe Interface) con questi parametri software
    OVERALL_PARAMETERS = ROS2ParamList(*RESPEAKER_PARAMETERS_ROS, *ADDITIONAL_ROS_PARAMETERS)
    
    NODE_NAME = "respeaker_node"

    def __init__(self):
        
        # Inizializza il nodo ROS standard
        super().__init__(self.NODE_NAME)

        self.robot_is_speaking=False

        # --- VARIABILI DI STATO ---
        self._is_vad_active = True # Se False, il nodo smette di processare la voce (utile per "zittire" il robot mentre parla lui)
        # Crea un servizio ROS per attivare/disattivare l'ascolto dinamicamente
        self._vad_active_service = self.create_service(SetBool, "/mic/active", self._vad_active_callback)
        
        # Registra una funzione da chiamare quando il nodo viene spento (per chiudere USB e Audio)
        self.context.on_shutdown(self.on_shutdown)
        
        # --- INIZIALIZZA INTERFACCIA HARDWARE (Controllo USB) ---
        self.respeaker = RespeakerInterface(self)

        # --- GESTIONE PARAMETRI ---
        self.config = None
        # Carica i valori iniziali dei parametri
        init_params = init_parameters(self, self.OVERALL_PARAMETERS)
        # Imposta una callback: se cambi un parametro live, chiama self.on_config
        self.add_on_set_parameters_callback(self.on_config)
        # Scrive i parametri iniziali dentro il chip hardware
        self.init_reaspeaker_parameters()
        
        # --- INIZIALIZZA STREAMING AUDIO (Cattura Suono) ---
        # Passiamo "self.on_audio": ogni volta che il microfono ha dati, chiamerà quella funzione
        self.respeaker_audio = RespeakerAudio(self, self.on_audio, suppress_error=self.suppress_pyaudio_error)
        
        # Buffer per accumulare la frase che l'utente sta dicendo ORA
        self.speech_audio_buffer = str() # Nota: idealmente dovrebbe essere b"" (bytes), ma Python gestisce la concatenazione
        # Flag: stiamo registrando una frase in questo momento?
        self.is_speeching = False
        # Timestamp dell'ultima volta che abbiamo sentito una voce
        self.speech_stopped = self.get_clock().now()
        
        # Variabili per memorizzare lo stato precedente (per pubblicare solo se cambia qualcosa)
        self.prev_is_voice = None
        self.prev_doa = None
        
        # --- CONFIGURAZIONE PUBLISHER (Topic in uscita) ---

        # Pubblica True/False se qualcuno sta parlando
        self.pub_vad = self.create_publisher(Bool, "is_speeching", qos_profile=LATCH_PROFILE)
        # Pubblica la direzione del suono in gradi (0-360)
        self.pub_doa_raw = self.create_publisher(Int32, "sound_direction", qos_profile=LATCH_PROFILE)
        # Pubblica la direzione come posa 3D per visualizzarla in RViz
        self.pub_doa = self.create_publisher(PoseStamped, "sound_localization", qos_profile=LATCH_PROFILE)
        # Pubblica l'audio continuo (streaming live)
        self.pub_audio = self.create_publisher(AudioData, "audio", 10)
        # Pubblica SOLO le frasi complete (questo è quello che usa Whisper/Google Speech)
        self.pub_speech_audio = self.create_publisher(AudioData, "speech_audio", 10)
        
        # Crea publisher individuali per ogni canale grezzo (debug)
        self.pub_audios = {
            c: self.create_publisher(AudioData, f"audio/channel{c}", 10)
            for c in self.respeaker_audio.channels
        }

        self.sub_robot_speaking = self.create_subscription(
            Bool,
            '/pepper/is_speaking',  # Topic che creeremo
            self.callback_robot_speaking,
            10
        )

        # --- SETUP BUFFER PREFETCH ---
        # Calcola quanti byte servono per salvare X secondi di audio (definito in speech_prefetch)
        # Formula: Secondi * Frequenza * (BitDepth / 8)
        self.speech_prefetch_bytes = int(
            self.speech_prefetch * self.respeaker_audio.rate * self.respeaker_audio.bitdepth / 8.0)
        self.speech_prefetch_buffer = b"" # Buffer circolare "memoria a breve termine"
        
        # Fa partire la registrazione audio
        self.respeaker_audio.start()
        
        # --- TIMER PRINCIPALE ---
        # Crea un timer che scatta 10 volte al secondo (10Hz) per eseguire self.on_timer
        self.info_timer = self.create_timer(1.0 / self.update_rate, self.on_timer)
        
        # Setup per il controllo dei LED tramite topic ROS
        self.timer_led = None
        self.sub_led = self.create_subscription(
                                                ColorRGBA,
                                                "status_led",
                                                self.on_status_led,
                                                qos_profile=10)

    # Callback per il servizio /mic/active (attiva/disattiva mic)
    def _vad_active_callback(self, req: SetBool.Request, resp: SetBool.Response):
        self._is_vad_active = req.data # Imposta lo stato
        # Resetta tutti i buffer per evitare rimasugli di audio vecchio
        self.speech_audio_buffer = str()
        self.speech_prefetch_buffer = b""
        self.is_speeching = False
        self.speech_stopped = rclpy.time.Time(clock_type=ClockType.ROS_TIME)
        self.prev_is_voice = None
        self.prev_doa = None    
        resp.success = self._is_vad_active
        return resp
    
    def callback_robot_speaking(self, msg):
        self.robot_is_speaking = msg.data
        if self.robot_is_speaking:
            self.get_logger().info("🚫 Robot parla: Input audio bloccato.")
        else:
            self.get_logger().info("👂 Robot finito: Input audio riattivato.")
    
    # Scrive i parametri ROS dentro il chip hardware all'avvio
    def init_reaspeaker_parameters(self):
        for p in RESPEAKER_PARAMETERS_ROS:
            name = p.name
            p_value = self.get_parameter(name).value
            self.respeaker.write(name, p_value) # Chiama USB Write

    # Pulizia quando il nodo muore
    def on_shutdown(self):
        try:
            self.respeaker.close() # Chiude connessione USB Controllo
        except:
            pass
        finally:
            self.respeaker = None
        try:
            self.respeaker_audio.stop() # Ferma stream Audio
        except:
            pass
        finally:
            self.respeaker_audio = None

    # Callback chiamata se cambi parametri live (es. rqt_reconfigure)
    def on_config(self, params: List[Parameter]):
        for p in params:
            # Se è un parametro software (es. timeout), aggiorna solo ROS
            if p in self.ADDITIONAL_ROS_PARAMETERS:
                defalut_reconfigure(self, [p], self.OVERALL_PARAMETERS)
            else:
                # Se è un parametro hardware (es. guadagno mic), scrivilo sul chip
                prev_val = self.respeaker.read(p.name)
                value = p.value
                if prev_val != value:
                    self.respeaker.write(p.name, value)
                    self.get_logger().warn(f"Reconfigure: {p.name}: {value}")

        return SetParametersResult(successful=True)

    # Callback chiamata quando arriva un messaggio su /status_led
    def on_status_led(self, msg):
        # Imposta il colore richiesto
        self.respeaker.set_led_color(r=msg.r, g=msg.g, b=msg.b, a=msg.a)
        
        # Se c'era un timer precedente per i LED, cancellalo
        if self.timer_led and self.timer_led.is_alive():
            self.timer_led.destroy()

        # Dopo 3 secondi, rimetti i LED in modalità "Trace" (ascolto normale)
        self.timer_led = self.create_timer(
            3.0,  # seconds
            lambda: self.respeaker.set_led_trace()
        )

    # --- CALLBACK AUDIO AD ALTA FREQUENZA (~93 volte/sec) ---
    # Viene chiamata dalla classe RespeakerAudio ogni volta che c'è un pacchetto dati
    def on_audio(self, data, channel):
        # Pubblica sempre l'audio grezzo sul topic del canale specifico
        self.pub_audios[channel].publish(AudioData(data=data))
        
        # Logica solo per il canale principale (quello usato per il riconoscimento)
        if channel == self.main_channel:
            # Pubblica stream continuo su /audio
            self.pub_audio.publish(AudioData(data=data))

            # 1. SE IL ROBOT PARLA: FLUSH E BLOCCO TOTALE
            if self.robot_is_speaking:
                # Svuotiamo il buffer di registrazione corrente
                self.speech_audio_buffer = b"" 
                
                # Svuotiamo il prefetch buffer (memoria a breve termine).
                # Questo è CRUCIALE: cancella la voce del robot dalla "memoria" del sistema.
                # Così quando torni ad ascoltare, il passato è vuoto.
                self.speech_prefetch_buffer = b""
                
                # (Opzionale) Se vuoi che VAD non scatti, puoi forzare is_speeching a False
                # self.is_speeching = False 
                
                return # Esce dalla funzione: ignora completamente questo pacchetto audio
            
            # SE IL SISTEMA HA RILEVATO VOCE ED È IN REGISTRAZIONE:
            if self.is_speeching:
                # Se è il primissimo pacchetto della frase:
                if len(self.speech_audio_buffer) == 0:
                    # Inserisci prima il "prefetch buffer".
                    # Serve a recuperare i suoni emessi APPENA PRIMA che il VAD scattasse
                    # (es. recupera la "C" di "Ciao" che altrimenti verrebbe tagliata)
                    self.speech_audio_buffer = self.speech_prefetch_buffer
                
                # Aggiungi i nuovi dati al buffer della frase corrente
                self.speech_audio_buffer += data
            
            # SE INVECE C'È SILENZIO:
            else:
                # Accumula i dati nel buffer circolare (memoria a breve termine)
                self.speech_prefetch_buffer += data
                # Mantieni il buffer della dimensione giusta (taglia i dati vecchi)
                self.speech_prefetch_buffer = self.speech_prefetch_buffer[-self.speech_prefetch_bytes:]

    # --- CICLO DI CONTROLLO PRINCIPALE (10 Hz) ---
    def on_timer(self):
        
        # Se siamo disattivati via servizio, non fare nulla
        if not self._is_vad_active:
            return
        
        stamp = self.get_clock().now()
        
        # --- LETTURA SENSORI VIA USB ---
        # !!! QUI È DOVE AVVIENE L'ERRORE USB TIMEOUT !!!
        # Chiede al chip: "Qualcuno sta parlando?"
        is_voice = self.respeaker.is_voice() 
        
        # Calcola la direzione del suono
        doa_rad = math.radians(self.respeaker.direction - 180.0)
        # Applica offset e correzioni geometriche
        doa_rad = angles.shortest_angular_distance(
            doa_rad, math.radians(self.doa_yaw_offset))
        doa = math.degrees(doa_rad)

        # --- GESTIONE VAD (Voice Activity Detection) ---
        # Se lo stato voce è cambiato (da Silenzio a Parlato o viceversa)
        if is_voice != self.prev_is_voice:
            # Pubblica il nuovo stato
            self.pub_vad.publish(Bool(data=bool(is_voice)))
            self.prev_is_voice = is_voice

        # --- GESTIONE DOA (Direction of Arrival) ---
        # Se la direzione è cambiata
        if doa != self.prev_doa:
            # Pubblica l'angolo grezzo (intero)
            self.pub_doa_raw.publish(Int32(data=int(doa)))
            self.prev_doa = doa

            # Costruisce un messaggio PoseStamped per visualizzare la freccia in RViz
            msg = PoseStamped()
            msg.header.frame_id = self.sensor_frame_id
            msg.header.stamp = stamp.to_msg()
            # Calcola quaternione per l'orientamento
            ori = T.quaternion_from_euler(math.radians(doa), 0, 0)
            # Calcola posizione X,Y
            msg.pose.position.x = self.doa_xy_offset * np.cos(doa_rad)
            msg.pose.position.y = self.doa_xy_offset * np.sin(doa_rad)
            msg.pose.orientation.w = ori[0]
            msg.pose.orientation.x = ori[1]
            msg.pose.orientation.y = ori[2]
            msg.pose.orientation.z = ori[3]
            self.pub_doa.publish(msg)

        # --- LOGICA DI SEGMENTAZIONE AUDIO (Il cuore del riconoscimento) ---
        
        # Se c'è voce adesso, aggiorna il timestamp "ultimo momento in cui ho sentito voce"
        if is_voice:
            self.speech_stopped = stamp
        
        # Calcola quanto tempo è passato dall'ultima volta che ho sentito voce
        diff = (stamp - self.speech_stopped).nanoseconds *1e-9 
        
        # CASO 1: C'è silenzio, MA è breve (meno di 2 secondi, 'speech_continuation')
        # Consideriamo che l'utente stia solo prendendo fiato -> Continua a registrare
        if diff < self.speech_continuation:
            self.is_speeching = True
        
        # CASO 2: C'è silenzio DA TROPPO TEMPO -> La frase è finita!
        elif self.is_speeching:
            buf = self.speech_audio_buffer # Prendi tutto l'audio accumulato
            self.speech_audio_buffer = str() # Pulisci il buffer
            self.is_speeching = False # Reset flag
            
            # Calcola durata in secondi per loggare
            duration = 8.0 * len(buf) * self.respeaker_audio.bitwidth
            duration = duration / self.respeaker_audio.rate / self.respeaker_audio.bitdepth
            self.get_logger().info("Speech detected for %.3f seconds" % duration)
            
            # Se la registrazione ha una durata sensata (non è un click o un errore)
            if self.speech_min_duration <= duration < self.speech_max_duration:
                # PUBBLICA L'AUDIO FINALE!
                # Questo messaggio viene catturato dal nodo audio_pipeline per essere trascritto
                self.pub_speech_audio.publish(AudioData(data=buf))

def main():
    if not rclpy.ok():
        rclpy.init()
    n = RespeakerNode()
    rclpy.spin(n)

if __name__ == '__main__':
    main()
