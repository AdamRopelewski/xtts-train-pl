"""Generate sample audio from a trained XTTS checkpoint using Coqui-TTS synthesize.

Example:
  python scripts/generate_sample.py \
    --text "Dzień dobry, to jest test." \
    --out generated.wav \
    --model-path xtts-base-model/xttsv2_2.0.2/model.pth \
    --config-path xtts-base-model/xttsv2_2.0.2/config.json \
    --use-cuda

This wraps `TTS.bin.synthesize.main` so it works without installing the `tts` CLI.
"""
import argparse


def main():
    # Dynamic default: enable CUDA if torch is available and CUDA is usable
    use_cuda_default = False
    try:
        import torch
        use_cuda_default = torch.cuda.is_available()
    except Exception:
        use_cuda_default = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Dzień dobry", help="Text to synthesize (default: 'Dzień dobry')")
    parser.add_argument("--out", default="run/samples/sample.wav", help="Output WAV path (default: run/samples/sample.wav)")
    parser.add_argument("--model-path", default="xtts-base-model/xttsv2_2.0.2", help="Path to model directory (default: xtts-base-model/xttsv2_2.0.2)")
    parser.add_argument("--config-path", default="xtts-base-model/xttsv2_2.0.2/config.json", help="Path to config.json (default: xtts-base-model/xttsv2_2.0.2/config.json)")
    parser.add_argument("--language_idx", default="pl", help="Language idx to pass to synthesize (default: pl)")
    parser.add_argument("--speaker-wav", default="train-data/speaker_reference/ref.wav", help="Path to speaker wav for external speaker voice (default: train-data/speaker_reference/ref.wav)")
    parser.add_argument("--use-cuda", action="store_true", default=use_cuda_default, help=f"Use CUDA for synthesis (default: {use_cuda_default})")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    args = parser.parse_args()

    # Ensure output dir exists
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Build arguments for the coqui synthesize entrypoint (match tts CLI flags)
    synth_args = [
        "--text", args.text,
        "--out_path", args.out,
        "--model_path", args.model_path,
        "--config_path", args.config_path,
        "--language_idx", str(args.language_idx),
    ]
    if args.use_cuda:
        synth_args += ["--use_cuda"]
    if args.speaker_wav:
        synth_args += ["--speaker_wav", args.speaker_wav]
    if args.no_progress:
        synth_args += ["--no_progress"]

    print("Running synth with:", " ".join(synth_args))

    # Import and call the coqui synthesize main
    try:
        from TTS.bin.synthesize import main as synth_main
    except Exception:
        print("Failed to import Coqui-TTS synthesize entrypoint. Ensure `coqui-tts` is installed/available in PYTHONPATH.")
        raise

    # Call the synthesize function
    synth_main(synth_args)


if __name__ == "__main__":
    main()
