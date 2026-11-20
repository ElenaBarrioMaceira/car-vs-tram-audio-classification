from pathlib import Path
import csv

import librosa
import numpy as np

project_root = Path(__file__).parent.parent

car_path = project_root / "data" / "raw" / "car"
tram_path = project_root / "data" / "raw" / "tram"

output_file = project_root / "data" / "processed" / "features.csv"
output_file.parent.mkdir(parents=True, exist_ok=True) # Ensure parent directory exists


def extract_features(audio_file, recording_user, label):

    y, sr = librosa.load(audio_file, sr=None)

    # Normalize audio
    y = librosa.util.normalize(y)

    # RMS (signal energy)
    rms = float(np.mean(librosa.feature.rms(y=y)))

    # Zero Crossing Rate
    zcr = float(
        np.mean(
            librosa.feature.zero_crossing_rate(y)
        )
    )

    # Calculate Spectral Centroid (sound brightness)
    spectral_centroid = float(
        np.mean(
            librosa.feature.spectral_centroid(
                y=y,
                sr=sr
            )
        )
    )

    # Calculate MFCC features
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    # Calculate mean value of each MFCC coefficient
    mfcc_means = np.mean(mfccs, axis=1)

    row = {
        "filename": audio_file.name,
        "recording_user": recording_user,
        "rms": rms,
        "zcr": zcr,
        "spectral_centroid": spectral_centroid,
        "label": label
    }

    # Add MFCC features to dictionary
    for i in range(13):
        row[f"mfcc_{i+1}"] = float(mfcc_means[i])

    return row


fieldnames = [
    "filename",
    "recording_user",
    "rms",
    "zcr",
    "spectral_centroid"
]

for i in range(1, 14):
    fieldnames.append(f"mfcc_{i}")

fieldnames.append("label")

# Create CSV file and write extracted features
with open(output_file, "w", newline="") as csv_file:

    # Create CSV writer and add column headers
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Process all WAV files in the car folder
    for audio_file in car_path.glob("*.wav"):
        print(f"Processing car: {audio_file.name}")

        row = extract_features(
            audio_file,
            recording_user="miika",
            label="car"
        )
        # Save extracted features to CSV
        writer.writerow(row)

    # Process all tram recordings
    for audio_file in tram_path.glob("*.wav"):
        print(f"Processing tram: {audio_file.name}")

        row = extract_features(
            audio_file,
            recording_user="Elena",
            label="tram"
        )
        # Save extracted features to CSV
        writer.writerow(row)

