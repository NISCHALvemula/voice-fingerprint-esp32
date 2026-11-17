# Voice fingerprint speaker identification

Speaker identification with a small pretrained embedding model from
github.com/tduffy000/ccny-masters-thesis, with the network also running
on an Arduino Nano ESP32.

## Desktop
pip install tensorflow numpy librosa sounddevice soundfile
Copy full_model.tflite (or quant_model.tflite) from the original repo into
this folder and rename it model.tflite.

python sid.py devices
python sid.py record --name NAME --session 1 --device N
python sid.py record --name NAME --session 2 --set test --device N
python sid.py enroll
python sid.py confusion --threshold 0.0 --margin 0

## Arduino Nano ESP32
Install the ArduTFLite library in the Arduino IDE.
python make_header.py model.tflite
Upload esp32_embedder.ino, then:
pip install pyserial
python bridge.py check --port COM5
python bridge.py confusion --port COM5

Recordings are not included because voice is personal data.
