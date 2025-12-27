import torch

model_dir = "run/training/GPT_XTTS_v2.0_LJSpeech_FT-Przelecz-v1-December-20-2025_11+58PM-0000000/"
# model_path = model_dir + "best_model.pth"
model_path = model_dir + "checkpoint_6600.pth"
checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

del checkpoint["optimizer"]

# https://github.com/coqui-ai/TTS/discussions/3474#discussioncomment-7965683
for key in list(checkpoint["model"].keys()):
    if "dvae" in key:
        del checkpoint["model"][key]

torch.save(checkpoint, model_dir+"model_small.pth")
