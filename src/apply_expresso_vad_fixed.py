import argparse
from pathlib import Path
import pydub
import sys

def load_vad_segments(vad_file):
    vad_segments = {}  # {filename: {channel: [(start, end)]}}
    
    with open(vad_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line or 'longform' in line:
                continue
                
            filename_with_channel, times_str = line.split('\t')
            filename, channel = filename_with_channel.split('/')
            
            if filename not in vad_segments:
                vad_segments[filename] = {}
                
            if channel in vad_segments[filename]:
                print(f"Warning: Duplicate channel {channel} for {filename}", file=sys.stderr)
                continue
                
            vad_segments[filename][channel] = [
                (float(start), float(end))
                for part in times_str.strip().split(') ')
                if part and (start := part.replace('(', '').replace(')', '').split(', ')[0])
                and (end := part.replace('(', '').replace(')', '').split(', ')[1])
            ]
    
    return vad_segments

def process_audio_file(wav_file, vad_segments, output_dir):
    filename = wav_file.stem
    if filename not in vad_segments:
        print(f"Warning: VAD segments not found for {filename}", file=sys.stderr)
        return
    
    # load before mkdir so we don't create empty dirs for broken files
    try:
        audio = pydub.AudioSegment.from_wav(str(wav_file))  # str() avoids Path issues on Windows
    except Exception as e:
        print(f"Error loading {wav_file}: {e}", file=sys.stderr)
        return

    if len(audio) == 0:
        print(f"Warning: Empty audio in {wav_file}, skipping", file=sys.stderr)
        return

    if audio.channels < 2:
        print(f"Warning: Expected stereo, got {audio.channels} channel(s) in {wav_file}, skipping", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    mono_channels = audio.split_to_mono()  # split once, reuse
    channels = {
        'channel1': mono_channels[0],
        'channel2': mono_channels[1]
    }
    
    for channel_name, audio_channel in channels.items():
        if channel_name not in vad_segments[filename]:
            print(f"Warning: No VAD segments for {filename}/{channel_name}", file=sys.stderr)
            continue

        for start, end in vad_segments[filename][channel_name]:
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)

            # guard against VAD timestamps exceeding actual audio length
            if start_ms >= len(audio_channel):
                print(f"Warning: Segment {start}-{end}s out of range for {filename}/{channel_name}, skipping", file=sys.stderr)
                continue

            segment = audio_channel[start_ms:end_ms]
            segment_path = output_dir / f"{filename}_{channel_name}_segment_{start}_{end}.wav"
            segment.export(segment_path, format="wav")

def main():
    parser = argparse.ArgumentParser(description="Apply VAD segmentation to Expresso audio files")
    parser.add_argument('expresso_root', type=Path, help='Root directory of Expresso dataset')
    args = parser.parse_args()
    
    vad_file = args.expresso_root / "VAD_segments.txt"
    input_dir = args.expresso_root / "audio_48khz" / "conversational"
    output_dir = args.expresso_root / "audio_48khz" / "conversational_vad_segmented"
    
    if not vad_file.is_file():
        print(f"Error: VAD segments file not found at {vad_file}", file=sys.stderr)
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"Error: Input directory not found at {input_dir}", file=sys.stderr)
        sys.exit(1)
        
    vad_segments = load_vad_segments(vad_file)
    print("Processing audio files...")
    
    for wav_file in input_dir.glob('**/*.wav'):
        relative_path = wav_file.relative_to(input_dir)
        output_subdir = output_dir / relative_path.parent
        process_audio_file(wav_file, vad_segments, output_subdir)

if __name__ == '__main__':
    main()