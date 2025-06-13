import argparse
import os
import socket
import tempfile
import time
import wave

import httpx
import numpy as np
import openai
import sounddevice as sd
from openai import OpenAIError
from pydub import AudioSegment
from pydub.playback import play


def call_with_retry(call_fn, max_retries=3, timeout_sec=10, wait_sec=3):
    for attempt in range(max_retries):
        try:
            return call_fn(timeout=timeout_sec)
        except (
            openai.APITimeoutError,
            httpx.TimeoutException,
            httpx.RequestError,
        ) as e:
            print(f"[Retry] Attempt {attempt+1} failed with timeout: {e}")
            time.sleep(wait_sec)
        except OpenAIError as e:
            print(f"[Error] OpenAI API error: {e}")
            break
    raise RuntimeError(f"Failed after {max_retries} attempts.")


class VoiceRobotAssistant:
    def __init__(self, mic_name, keep_tmp, no_speech=False):
        self.mic_name = mic_name
        self.keep_tmp = keep_tmp
        self.no_speech = no_speech

        self.samplerate = 16000
        self.frame_duration_ms = 50
        self.silence_timeout = 1.2
        self.energy_threshold = 15.0

        self.tmpdir = None
        self.tmpdir_obj = None
        self.wav_path = None
        self.tts_path = None

        self.first_prompt = True

        openai.api_key = os.environ["OPENAI_API_KEY"]

        self.device_index = self.select_input_device()
        print(f"🎙️ Selected microphone: {self.device_index}.")

    def select_input_device(self):
        print(sd.query_devices())
        for idx, device in enumerate(sd.query_devices()):
            if self.mic_name in device["name"] and device["max_input_channels"] > 0:
                return idx
        raise RuntimeError(f"Microphone '{self.mic_name}' not found.")

    def setup_tmpdir(self):
        if self.keep_tmp:
            self.tmpdir_obj = tempfile.TemporaryDirectory()
            self.tmpdir = self.tmpdir_obj.name
            return None
        else:
            tmpdir_obj = tempfile.TemporaryDirectory()
            self.tmpdir = tmpdir_obj.name
            return tmpdir_obj

    def record_until_silence(self):
        frame_size = int(self.samplerate * self.frame_duration_ms / 1000)
        silence_frames = int(self.silence_timeout * 1000 / self.frame_duration_ms)
        silence_counter = 0
        recorded = []
        started = False
        stop_flag = False

        time.sleep(0.2)
        print("📣 Waiting for speech... (Start speaking to begin recording)")

        def audio_callback(indata, frames, time_, status):
            nonlocal recorded, silence_counter, started, stop_flag
            energy = np.linalg.norm(indata)
            print(f"Energy: {energy:.2f}", end="\r", flush=True)

            if not started and energy > self.energy_threshold:
                print("🎙️ Speech detected")
                started = True

            if started:
                recorded.append(indata.copy())
                if energy > self.energy_threshold:
                    silence_counter = 0
                else:
                    silence_counter += 1
                if silence_counter >= silence_frames:
                    stop_flag = True

        with sd.InputStream(
            device=self.device_index,
            channels=1,
            samplerate=self.samplerate,
            blocksize=frame_size,
            dtype="float32",
            callback=audio_callback,
        ):
            while not stop_flag:
                time.sleep(0.05)

        print("✅ Recording finished")
        audio = np.concatenate(recorded, axis=0)
        return audio

    def save_wav(self, audio):
        with wave.open(self.wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.samplerate)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

    def transcribe_audio(self):
        def api_call(timeout):
            with open(self.wav_path, "rb") as f:
                return openai.audio.transcriptions.create(
                    model="whisper-1", file=f, language="ja", timeout=timeout
                )

        transcript = call_with_retry(api_call)
        return transcript.text.strip()

    def find_closest_task_desc(self, user_text):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        task_path = os.path.join(script_dir, "task_desc.txt")
        with open(task_path, "r", encoding="utf-8") as f:
            task_list = [line.strip() for line in f if line.strip()]

        numbered_list = [f"{i}. {task}" for i, task in enumerate(task_list)]

        system_prompt = (
            "The following is a list of tasks that the robot can perform.\n"
            "Choose the task whose meaning is most similar to the user's request, and return only its number.\n"
            "Return only the integer number (e.g., 0 or 3).\n"
            "Interpret the request broadly and flexibly to find the best match.\n"
            "If no match can be found, respond with 'No matching task found'.\n\n"
            "Task list:\n" + "\n".join(numbered_list)
        )

        user_prompt = f"Request: {user_text}"

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        reply = response.choices[0].message.content.strip()
        try:
            selected_index = int(reply)
            return task_list[selected_index]
        except (ValueError, IndexError):
            raise RuntimeError(f"Invalid reply for task index: '{reply}'")

    def chat_with_gpt(self, task_desc):
        prompt = f"Respond politely in Japanese to the following command, using the format: 'わかりました。***をやります。' The command may include physical actions: {task_desc}"

        def api_call(timeout):
            return openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a kind Japanese-speaking robot assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                timeout=timeout,
            )

        response = call_with_retry(api_call)
        return response.choices[0].message.content.strip()

    def synthesize_speech(self, text):
        if self.no_speech:
            print(f"[Speech Skipped] Synthesized text: {text}")
            return

        def api_call(timeout):
            return openai.audio.speech.create(
                model="tts-1", voice="shimmer", input=text.strip(), timeout=timeout
            )

        response = call_with_retry(api_call)
        with open(self.tts_path, "wb") as f:
            f.write(response.content)

    def play_audio(self):
        if self.no_speech:
            print("[Speech Skipped] Playback skipped")
            return
        audio = AudioSegment.from_file(self.tts_path)
        play(audio)

    def send_task_to_robot(self, task_desc):
        SOCKET_PATH = "/tmp/rmb.sock"

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            print("[Comm] Connecting to robot...")
            client.connect(SOCKET_PATH)

            # Send task
            client.sendall(task_desc.encode("utf-8"))
            print(f"[Comm] Task sent: {task_desc}")

            # Wait for any speech from the human
            print("[Robot Action] Executing task. Please notify by voice when done.")
            audio = self.record_until_silence()
            self.save_wav(audio)
            _ = self.transcribe_audio()

            # Send 'done' signal
            client.sendall("done".encode("utf-8"))
            print("[Comm] Done signal sent")

            # Wait for response
            response = client.recv(1024).decode("utf-8")
            print(f"[Comm] Response from robot: {response}")

    def run_task_cycle(self, reprompt=True):
        if reprompt:
            if self.first_prompt:
                prompt_text = "何をしましょうか？"
                self.first_prompt = False
            else:
                prompt_text = "次は何をしましょうか？"

            print("[Robot]", prompt_text)
            self.synthesize_speech(prompt_text)
            self.play_audio()

        audio = self.record_until_silence()
        self.save_wav(audio)
        user_text = self.transcribe_audio()
        print("[Human]", user_text)

        try:
            task_desc = self.find_closest_task_desc(user_text)
        except RuntimeError as e:
            print(f"[Error] {e}")
            self.synthesize_speech(
                "すみません、分かりませんでした。もう一度言ってください。"
            )
            self.play_audio()
            return False

        print("[Task]", task_desc)

        reply_text = self.chat_with_gpt(task_desc)
        print("[Robot]", reply_text)
        self.synthesize_speech(reply_text)
        self.play_audio()

        self.send_task_to_robot(task_desc)
        return True

    def interaction_loop(self):
        tmpctx = self.setup_tmpdir()

        self.wav_path = os.path.join(self.tmpdir, "input.wav")
        self.tts_path = os.path.join(self.tmpdir, "speak.mp3")

        if tmpctx is not None:
            with tmpctx:
                while True:
                    success = self.run_task_cycle(reprompt=True)
                    while not success:
                        success = self.run_task_cycle(reprompt=False)
        else:
            while True:
                success = self.run_task_cycle(reprompt=True)
                while not success:
                    success = self.run_task_cycle(reprompt=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mic_name",
        type=str,
        default="USB PnP Audio Device",
        help="Microphone device name",
    )
    parser.add_argument(
        "--keep_tmp",
        action="store_true",
        help="Keep temporary directory after execution",
    )
    parser.add_argument(
        "--no_speech", action="store_true", help="Do not synthesize or play audio"
    )
    args = parser.parse_args()

    assistant = VoiceRobotAssistant(
        mic_name=args.mic_name, keep_tmp=args.keep_tmp, no_speech=args.no_speech
    )
    assistant.interaction_loop()
