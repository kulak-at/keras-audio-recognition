from pydub import AudioSegment

sound1 = AudioSegment.from_wav("./data/bed/0a7c2a8d_nohash_0.wav")
sound2 = AudioSegment.from_wav("./data/__laptop_noise/1525040672.6543267.wav")

output = sound1.overlay(sound2 + -5)

output.export("./exported.wav", format="wav")
