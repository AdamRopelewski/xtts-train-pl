import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import (
    GPTArgs,
    GPTTrainer,
    GPTTrainerConfig,
)
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager






# Logging parameters
RUN_NAME = "GPT_XTTS_v2.0_LJSpeech_FT-Przelecz-v2"
PROJECT_NAME = "XTTS_trainer-Przelecz-v2"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

# Set here the path that the checkpoints will be saved. Default: ./run/training/
OUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "run", "training"
)

# Training Parameters
OPTIMIZER_WD_ONLY_ON_WEIGHTS = (
    True  # for multi-gpu training please make it False
)
START_WITH_EVAL = True  # if True it will star with evaluation
BATCH_SIZE = 2  # set here the batch size
GRAD_ACUMM_STEPS = 126  # set here the grad accumulation steps
EPOCHS = 1000  # set here the number of epochs
WARMUP_EPOCHS = 1  # set here the number of warmup epochs
# Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least 252 for more efficient training.
# You can increase/decrease BATCH_SIZE but then set GRAD_ACUMM_STEPS accordingly.
PRINT_STEP = 455
PLOT_STEP = 455
SAVE_STEP = 4555




# Define here the dataset that you want to use for the fine-tuning on.
config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="ljspeech-przelecz-v1",
    path="train-data",
    meta_file_train="csv/metadata.csv",
    language="pl",
)

# Add here the configs of the datasets
DATASETS_CONFIG_LIST = [config_dataset]

# Set the path to the downloaded files
DVAE_CHECKPOINT = "xtts-base-model/xttsv2_2.0.2/dvae.pth"
MEL_NORM_FILE = "xtts-base-model/xttsv2_2.0.2/mel_stats.pth"

if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print("no DVAE files found")
    exit(1)


TOKENIZER_FILE = "xtts-base-model/xttsv2_2.0.2/vocab.json"
XTTS_CHECKPOINT = "xtts-base-model/xttsv2_2.0.2/model.pth"


if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print("no XTTS v2.0 files found")
    exit(1)


# Training sentences generations
SPEAKER_REFERENCE = [
    # speaker reference to be used in training test sentences
    "train-data/speaker_reference/zoladkowicz-przelecz-v1-ref-v2.wav"
]
LANGUAGE = config_dataset.language


def main():
    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,  # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        # checkpoint path of the model that you want to fine-tune
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # define audio config
    audio_config = XttsAudioConfig(
        sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000
    )
    # training parameters config
    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="""
            GPT XTTS training
            """,
        dashboard_logger=DASHBOARD_LOGGER,
        epochs=EPOCHS,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        # batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=2,
        eval_split_max_size=256,
        print_step=PRINT_STEP,
        plot_step=PLOT_STEP,
        log_model_step=1000,
        save_step=SAVE_STEP,
        save_n_checkpoints=EPOCHS-WARMUP_EPOCHS,  # save a checkpoint every epoch
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=True,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={
            "betas": [0.9, 0.96],
            "eps": 1e-8,
            "weight_decay": 1e-2,
        },
        lr=1e-06,  # learning rate
        lr_scheduler="MultiStepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={
            "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
            "gamma": 0.5,
            "last_epoch": -1
        },
        test_sentences=[
            {
                "text": "Dość długo zajęło mi znalezienie własnego głosu, a teraz, gdy już go mam, nie zamierzam milczeć.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "To ciasto jest wspaniałe. Jest takie pyszne i wilgotne.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "Przygotuj bigos, zaczynając od podsmażenia drobno posiekanych onions na cooking oil, aż zaczną rizzować jak sigma male, tworząc gyatt aromat. Dodaj pokrojone pork, beef oraz kiełbasę, smażąc wszystko intensywnie",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "Szczepan Szczygieł, Z Grzmiących Bystrzyc, Przed chrzcinami,Chciał się przystrzyc. Sam się strzyc, Nie przywykł wszakże, Więc do szwagra, Skoczył: Szwagrze!",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "13 marca 1966 roku nasz projekt przekroczył 50% ukończenia i zajął 1 miejsce.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
        ],
    )

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
