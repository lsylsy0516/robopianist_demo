#!/usr/bin/env python3
"""MIDI Alignment Evaluation Script

This script evaluates robot MIDI performance by aligning it with a reference MIDI file
using Dynamic Time Warping (DTW) algorithm, with optional audio similarity scoring.
"""

import pretty_midi
import librosa
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine
from typing import Tuple, List, Optional


def extract_audio_from_video(video_path: str, sr: int = 44100) -> Optional[np.ndarray]:
    """Extract audio waveform from video file.
    
    Args:
        video_path: Path to video file (e.g., MP4).
        sr: Target sample rate.
        
    Returns:
        Audio waveform array, or None if extraction fails.
    """
    try:
        # librosa can directly load audio from video files
        audio, _ = librosa.load(video_path, sr=sr, mono=True)
        return audio
    except Exception as e:
        print(f"Warning: Could not extract audio from video {video_path}: {e}")
        return None


def synthesize_midi_audio(midi_path: str, sr: int = 44100) -> Optional[np.ndarray]:
    """Synthesize audio waveform from MIDI file using robopianist music module.
    
    Args:
        midi_path: Path to MIDI file.
        sr: Target sample rate.
        
    Returns:
        Audio waveform array, or None if synthesis fails.
    """
    try:
        # Try to use robopianist's music.load and synthesize
        from robopianist import music
        midi_file = music.load(midi_path)
        audio = midi_file.synthesize(sampling_rate=sr)
        return audio
    except ImportError:
        # Fallback to pretty_midi's fluidsynth if robopianist is not available
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            # Use fluidsynth for better quality
            audio = midi.fluidsynth(fs=sr)
            return audio
        except Exception as e:
            print(f"Warning: Could not synthesize MIDI {midi_path}: {e}")
            return None
    except Exception as e:
        print(f"Warning: Could not synthesize MIDI {midi_path}: {e}")
        return None


def compute_audio_similarity(
    audio1: np.ndarray,
    audio2: np.ndarray,
    sr: int = 44100,
    hop_length: int = 512
) -> float:
    """Compute cosine similarity between two audio signals using MFCC features.
    
    Args:
        audio1: First audio waveform.
        audio2: Second audio waveform.
        sr: Sample rate.
        hop_length: Hop length for MFCC computation.
        
    Returns:
        Cosine similarity score between 0.0 and 1.0 (higher is better).
    """
    try:
        # Compute MFCC features for both audio signals
        mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr, hop_length=hop_length, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr, hop_length=hop_length, n_mfcc=13)
        
        # Average across time to get fixed-size feature vectors
        mfcc1_mean = np.mean(mfcc1, axis=1)
        mfcc2_mean = np.mean(mfcc2, axis=1)
        
        # Compute cosine similarity (1 - cosine distance)
        # Cosine similarity ranges from -1 to 1, we normalize to 0 to 1
        cosine_similarity = 1.0 - cosine(mfcc1_mean, mfcc2_mean)
        cosine_similarity = max(0.0, cosine_similarity)  # Clip to [0, 1]
        
        return cosine_similarity
    except Exception as e:
        print(f"Warning: Could not compute audio similarity: {e}")
        return 0.0


def extract_note_features(notes: List[pretty_midi.Note]) -> np.ndarray:
    """Extract note features: [start_time, pitch, duration].
    
    Args:
        notes: List of PrettyMIDI note objects.
        
    Returns:
        N x 3 array where each row is [start_time, pitch, duration].
    """
    features = []
    for note in notes:
        features.append([note.start, note.pitch, note.end - note.start])
    return np.array(features)


def compute_cost_matrix(ref_features: np.ndarray, perf_features: np.ndarray) -> np.ndarray:
    """Compute cost matrix between reference and performance features.
    
    Cost function heavily penalizes pitch differences:
    cost = 100 * |pitch_diff| + |start_diff| + |duration_diff|
    
    Args:
        ref_features: N x 3 array of reference note features.
        perf_features: M x 3 array of performance note features.
        
    Returns:
        N x M cost matrix.
    """
    n_ref = len(ref_features)
    n_perf = len(perf_features)
    cost_matrix = np.zeros((n_ref, n_perf))
    
    for i in range(n_ref):
        for j in range(n_perf):
            pitch_diff = abs(ref_features[i, 1] - perf_features[j, 1])
            start_diff = abs(ref_features[i, 0] - perf_features[j, 0])
            duration_diff = abs(ref_features[i, 2] - perf_features[j, 2])
            
            # Heavily penalize pitch differences
            cost = 100.0 * pitch_diff + start_diff + duration_diff
            cost_matrix[i, j] = cost
    
    return cost_matrix


def calculate_alignment_score(
    reference_midi_path: str,
    performed_midi_path: str,
    video_path: Optional[str] = None,
    use_audio_similarity: bool = False,
    audio_weight: float = 0.2
) -> float:
    """Calculate alignment score between reference and performed MIDI files.
    
    This function uses DTW to align the two MIDI files and computes a reward score
    based on how well the performance matches the reference. Optionally incorporates
    audio similarity between video audio and synthesized MIDI audio.
    
    Args:
        reference_midi_path: Path to reference MIDI file.
        performed_midi_path: Path to performed MIDI file.
        video_path: Optional path to video file with audio track (e.g., MP4).
        use_audio_similarity: Whether to incorporate audio similarity scoring.
        audio_weight: Weight for audio similarity (0.0-1.0, default 0.2).
        
    Returns:
        Alignment score between 0.0 and 1.0 (higher is better).
        
    Raises:
        ValueError: If MIDI file has no instruments or notes.
    """
    # Load MIDI files
    ref_midi = pretty_midi.PrettyMIDI(reference_midi_path)
    perf_midi = pretty_midi.PrettyMIDI(performed_midi_path)
    
    # Extract notes from first instrument
    if len(ref_midi.instruments) == 0:
        raise ValueError("Reference MIDI has no instruments")
    if len(perf_midi.instruments) == 0:
        raise ValueError("Performed MIDI has no instruments")
    
    ref_notes = ref_midi.instruments[0].notes
    perf_notes = perf_midi.instruments[0].notes
    
    if len(ref_notes) == 0:
        raise ValueError("Reference MIDI has no notes")
    if len(perf_notes) == 0:
        # If performance has no notes, return 0 score
        return 0.0
    
    # Extract note features
    ref_features = extract_note_features(ref_notes)
    perf_features = extract_note_features(perf_notes)
    
    # Compute cost matrix
    cost_matrix = compute_cost_matrix(ref_features, perf_features)
    
    # Run DTW alignment
    try:
        D, wp = librosa.sequence.dtw(
            C=cost_matrix,
            step_sizes_sigma=np.array([[1, 1], [0, 1], [1, 0]])
        )
    except Exception as e:
        print(f"DTW alignment failed: {e}")
        return 0.0
    
    # Initialize penalty tracking
    total_penalty = 0.0
    aligned_ref_indices = set()
    aligned_perf_indices = set()
    
    # Traverse DTW path and accumulate penalties
    for path_idx in range(len(wp)):
        ref_idx = wp[path_idx, 0]
        perf_idx = wp[path_idx, 1]
        
        ref_note = ref_notes[ref_idx]
        perf_note = perf_notes[perf_idx]
        
        # Calculate pitch penalty (heavy penalty for wrong pitch)
        if ref_note.pitch != perf_note.pitch:
            pitch_penalty = 100.0 * abs(ref_note.pitch - perf_note.pitch)
            total_penalty += pitch_penalty
        
        # Calculate start time difference penalty
        start_penalty = abs(ref_note.start - perf_note.start)
        total_penalty += start_penalty
        
        # Calculate duration difference penalty
        duration_penalty = abs((ref_note.end - ref_note.start) - (perf_note.end - perf_note.start))
        total_penalty += duration_penalty
        
        # Track aligned notes
        aligned_ref_indices.add(ref_idx)
        aligned_perf_indices.add(perf_idx)
    
    # Calculate penalty for missed and extra notes
    missed_notes = len(ref_notes) - len(aligned_ref_indices)
    extra_notes = len(perf_notes) - len(aligned_perf_indices)
    
    # Add heavy penalty for missed/extra notes (these weren't aligned by DTW)
    missing_penalty = missed_notes * 500.0  # Large penalty per missed note
    extra_penalty = extra_notes * 200.0     # Penalty per extra note (less than missing)
    total_penalty += missing_penalty + extra_penalty
    
    # Convert total penalty to normalized reward score (0-1, higher is better)
    # Normalize by expected penalty: len(ref_notes) * 100
    normalization_factor = len(ref_notes) * 100.0
    if normalization_factor == 0:
        reward = 1.0  # If no reference notes, consider it perfect
    else:
        reward = max(0.0, 1.0 - total_penalty / normalization_factor)
    
    # Optionally incorporate audio similarity if requested
    if use_audio_similarity:
        try:
            if video_path is not None:
                # Extract audio from video
                video_audio = extract_audio_from_video(video_path)
                
                if video_audio is not None:
                    # Synthesize reference MIDI audio
                    ref_audio = synthesize_midi_audio(reference_midi_path)
                    
                    if ref_audio is not None:
                        # Compute audio similarity score
                        audio_score = compute_audio_similarity(ref_audio, video_audio)
                        
                        # Combine alignment score and audio similarity
                        # audio_weight: how much to weight audio similarity (default 0.2 = 20%)
                        reward = (1.0 - audio_weight) * reward + audio_weight * audio_score
                        
                        print(f"Audio similarity: {audio_score:.4f}")
                    else:
                        print("Warning: Could not synthesize reference MIDI audio")
                else:
                    print("Warning: Could not extract audio from video")
            else:
                print("Warning: Video path not provided, skipping audio similarity")
        except Exception as e:
            print(f"Warning: Audio similarity calculation failed: {e}")
            # Continue with alignment score only
    
    return reward


if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description="Evaluate robot performance by comparing video audio with reference MIDI"
    )
    parser.add_argument("video_path", type=str, help="Path to video file with audio track (e.g., MP4)")
    parser.add_argument("performed_midi_path", type=str, help="Path to performed MIDI file from the video")
    parser.add_argument(
        "--reference-midi",
        type=str,
        default=None,
        help="Path to reference MIDI file (default: twinkle-twinkle-trimmed.mid)"
    )
    parser.add_argument(
        "--audio-weight",
        type=float,
        default=0.2,
        help="Weight for audio similarity (0.0-1.0, default 0.2)"
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio similarity scoring (use only DTW alignment)"
    )
    
    args = parser.parse_args()
    
    # Set up file paths
    script_dir = Path(__file__).parent
    
    # Use provided reference file or default
    if args.reference_midi:
        ref_file = Path(args.reference_midi)
    else:
        ref_file = script_dir / 'twinkle-twinkle-trimmed.mid'
    
    video_file = Path(args.video_path)
    perf_file = Path(args.performed_midi_path)
    
    print("=== MIDI Alignment Evaluation ===\n")
    print(f"Reference MIDI: {ref_file.name}")
    print(f"Performed MIDI: {perf_file.name}")
    print(f"Video: {video_file.name}")
    
    # Check if files exist
    if not ref_file.exists():
        print(f"Error: Reference file not found: {ref_file}")
        sys.exit(1)
    
    if not perf_file.exists():
        print(f"Error: Performed MIDI not found: {perf_file}")
        sys.exit(1)
    
    if not video_file.exists():
        print(f"Error: Video file not found: {video_file}")
        sys.exit(1)
    
    # Evaluate with audio similarity by default (unless --no-audio is specified)
    use_audio = not args.no_audio
    
    # Try to evaluate
    try:
        score = calculate_alignment_score(
            str(ref_file),
            str(perf_file),
            video_path=str(video_file),
            use_audio_similarity=use_audio,
            audio_weight=args.audio_weight
        )
        print(f"\nAlignment Score: {score:.4f}")
        print(f"  - Score range: 0.0 (worst) to 1.0 (best)")
        
        if score > 0.8:
            print(f"  - Quality: Excellent ✓")
        elif score > 0.6:
            print(f"  - Quality: Good")
        elif score > 0.4:
            print(f"  - Quality: Fair")
        else:
            print(f"  - Quality: Poor")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

