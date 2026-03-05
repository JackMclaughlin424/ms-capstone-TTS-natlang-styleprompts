"""
Adapted from Diwan et al.'s apply_expresso_vad.py, fixed bugs, added logging and multithreading.
"""
import argparse
from pathlib import Path
import pydub
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# lock so concurrent workers don't interleave their stderr warnings
_print_lock = threading.Lock()

def warn(msg):
    with _print_lock:
        print(msg, file=sys.stderr)

def process_audio_file(wav_file, vad_segments, output_dir):
    filename = wav_file.stem
    if filename not in vad_segments:
        warn(f"Warning: VAD segments not found for {filename}")
        return 0
    
    # load before mkdir so we don't create empty dirs for broken files
    try:
        audio = pydub.AudioSegment.from_wav(str(wav_file))  # str() avoids Path issues on Windows
    except Exception as e:
        warn(f"Error loading {wav_file}: {e}")
        return 0

    if len(audio) == 0:
        warn(f"Warning: Empty audio in {wav_file}, skipping")
        return 0

    if audio.channels < 2:
        warn(f"Warning: Expected stereo, got {audio.channels} channel(s) in {wav_file}, skipping")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    mono_channels = audio.split_to_mono()  # split once, reuse
    channels = {
        'channel1': mono_channels[0],
        'channel2': mono_channels[1]
    }

    segments_written = 0
    for channel_name, audio_channel in channels.items():
        if channel_name not in vad_segments[filename]:
            warn(f"Warning: No VAD segments for {filename}/{channel_name}")
            continue

        for start, end in vad_segments[filename][channel_name]:
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)

            # guard against VAD timestamps exceeding actual audio length
            if start_ms >= len(audio_channel):
                warn(f"Warning: Segment {start}-{end}s out of range for {filename}/{channel_name}, skipping")
                continue

            segment = audio_channel[start_ms:end_ms]
            segment_path = output_dir / f"{filename}_{channel_name}_segment_{start}_{end}.wav"
            segment.export(segment_path, format="wav")
            segments_written += 1

    return segments_written

def main():
    parser = argparse.ArgumentParser(description="Apply VAD segmentation to Expresso audio files")
    parser.add_argument('expresso_root', type=Path, help='Root directory of Expresso dataset')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel worker threads (default: 8)')
    args = parser.parse_args()
    
    vad_file = args.expresso_root / "VAD_segments.txt"
    input_dir = args.expresso_root / "audio_48khz" / "conversational"
    output_dir = args.expresso_root / "audio_48khz" / "conversational_vad_segmented_multi"
    
    if not vad_file.is_file():
        print(f"Error: VAD segments file not found at {vad_file}", file=sys.stderr)
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"Error: Input directory not found at {input_dir}", file=sys.stderr)
        sys.exit(1)
        
    vad_segments = load_vad_segments(vad_file)

    wav_files = list(input_dir.glob('**/*.wav'))
    total_files = len(wav_files)
    print(f"Found {total_files} file(s), processing with {args.workers} workers...")

    completed = 0
    total_segments = 0

    def submit(wav_file):
        relative_path = wav_file.relative_to(input_dir)
        output_subdir = output_dir / relative_path.parent
        return process_audio_file(wav_file, vad_segments, output_subdir)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(submit, f): f for f in wav_files}
        for future in as_completed(futures):
            wav_file = futures[future]
            completed += 1
            try:
                segments_written = future.result()
                total_segments += segments_written
            except Exception as e:
                # shouldn't normally reach here since process_audio_file catches internally
                warn(f"Error processing {wav_file}: {e}")
                segments_written = 0
            # \r so we reuse the line instead of spamming
            print(f"  [{completed}/{total_files}] {wav_file.name} -> {segments_written} segment(s)", end='\r' if completed < total_files else '\n')

    print(f"Done. {total_segments} segment(s) written to {output_dir}")

if __name__ == '__main__':
    main()