import os
import json
import base64
import queue
import threading
import time
import websocket
import pyaudio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -------------------
# Configuration
# -------------------
API_KEY = os.getenv("OPENAI_API_KEY")
WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime"

CHUNK_SIZE = 1024
RATE = 24000
FORMAT = pyaudio.paInt16

audio_buffer = bytearray()
mic_queue = queue.Queue()
stop_event = threading.Event()

# -------------------
# Microphone Callback
# -------------------
def mic_callback(in_data, frame_count, time_info, status):
    mic_queue.put(in_data)
    return (bytes(in_data), pyaudio.paContinue)  # <-- FIXED

# -------------------
# Speaker Callback
# -------------------
def speaker_callback(in_data, frame_count, time_info, status):
    bytes_needed = frame_count * 2
    global audio_buffer

    if len(audio_buffer) >= bytes_needed:
        chunk = audio_buffer[:bytes_needed]
        audio_buffer = audio_buffer[bytes_needed:]
    else:
        chunk = audio_buffer + b'\x00' * (bytes_needed - len(audio_buffer))
        audio_buffer.clear()

    return (bytes(chunk), pyaudio.paContinue)  # <-- FIXED

# -------------------
# Send Mic Audio to OpenAI
# -------------------
def send_mic_audio(ws):
    while not stop_event.is_set():
        if not mic_queue.empty():
            chunk = mic_queue.get()
            encoded = base64.b64encode(chunk).decode("utf-8")
            message = {"type": "input_audio_buffer.append", "audio": encoded}
            try:
                ws.send(json.dumps(message))
            except Exception as e:
                print("Error sending mic audio:", e)
        else:
            time.sleep(0.01)

# -------------------
# Receive Audio from OpenAI
# -------------------
def receive_audio(ws):
    global audio_buffer
    while not stop_event.is_set():
        try:
            msg = ws.recv()
            if not msg:
                break
            event = json.loads(msg)

            if event.get("type", "") == "response.audio.delta":
                audio_buffer.extend(base64.b64decode(event["delta"]))

        except Exception as e:
            print("Error receiving audio:", e)

# -------------------
# Send Session Config
# -------------------
def send_session_config(ws):
    config = {
        "type": "session.update",
        "session": {
            "instructions": "You are a friendly AI assistant. Always respond in English only.Only answer my questions if they are in English. Ignore any other language.",
            "voice": "alloy",
            "modalities": ["text", "audio"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": {"type": "server_vad", "threshold": 0.5},
        }
    }
    ws.send(json.dumps(config))

# -------------------
# Connect to OpenAI
# -------------------
def connect_to_openai():
    ws = websocket.create_connection(
        WS_URL,
        header=[
            f"Authorization: Bearer {API_KEY}",
            "OpenAI-Beta: realtime=v1"
        ]
    )
    print("Connected to OpenAI Realtime API.")

    send_session_config(ws)

    threading.Thread(target=receive_audio, args=(ws,), daemon=True).start()
    threading.Thread(target=send_mic_audio, args=(ws,), daemon=True).start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        ws.close()
        print("WebSocket closed.")

# -------------------
# Main
# -------------------
def main():
    p = pyaudio.PyAudio()

    mic_stream = p.open(format=FORMAT, channels=1, rate=RATE, input=True,
                        frames_per_buffer=CHUNK_SIZE, stream_callback=mic_callback)

    speaker_stream = p.open(format=FORMAT, channels=1, rate=RATE, output=True,
                            frames_per_buffer=CHUNK_SIZE, stream_callback=speaker_callback)

    mic_stream.start_stream()
    speaker_stream.start_stream()

    connect_to_openai()

    mic_stream.stop_stream()
    mic_stream.close()
    speaker_stream.stop_stream()
    speaker_stream.close()
    p.terminate()
    print("Audio streams closed. Exiting.")

if __name__ == "__main__":
    main()
