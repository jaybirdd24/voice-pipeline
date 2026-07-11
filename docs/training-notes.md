# Training Notes

Every `train_piper.sh` run — smoke or production, success or failure — appends
an entry under **Runs** below (written by the script itself). Don't edit old
entries; add corrections as new ones.

## Decisions

**espeak voice for English (2026-07-10): keep `en-GB-x-rp`.**
espeak-ng ships no `en-AU`/Australian English voice — confirmed in both the
system install and Piper's bundled espeak-ng-data (available English voices:
`en-gb`, `en-us`, `en-gb-x-rp`, `en-gb-scotland`, `en-gb-x-gbclan`,
`en-gb-x-gbcwmd`, `en-029`). `en-GB-x-rp` is phonemically the closest to
en-AU, every prior successful en-AU-accent fine-tune in the p4p workspace used
it, and the accent is carried by the cloned training audio, not the
phonemizer. Revisit only if AU-specific vowels sound wrong in evaluation.

## Runs

## 2026-07-11 17:49 NZST — en (smoke)
- dataset: /home/jay/p4p/datasets/en_qwen3_synth (116 clips)
- base checkpoint: /home/jay/p4p/piper_training/pretrained/alan_medium/en/en_GB/alan/medium/alan-medium-stripped.ckpt (pretrained base checkpoint)
- espeak voice: en-GB-x-rp
- hyperparams: batch=8 accum=1 lr=0.00002 lr_d=0.00001 max_epochs=6340 sample_rate=22050
- output dir: /home/jay/p4p/piper_training/training_output_en_qwen3_synth
- outcome: success — final ckpt: <none found>

## 2026-07-11 17:52 NZST — en (smoke)
- dataset: /home/jay/p4p/datasets/en_qwen3_synth (116 clips)
- base checkpoint: /home/jay/p4p/piper_training/pretrained/alan_medium/en/en_GB/alan/medium/alan-medium-stripped.ckpt (pretrained base checkpoint)
- espeak voice: en-GB-x-rp
- hyperparams: batch=8 accum=1 lr=0.00002 lr_d=0.00001 max_epochs=6341 sample_rate=22050
- output dir: /home/jay/p4p/piper_training/training_output_en_qwen3_synth
- outcome: success — final ckpt: /home/jay/p4p/piper_training/training_output_en_qwen3_synth/lightning_logs/version_1/checkpoints/epoch=6340-step=26.ckpt

## 2026-07-11 17:53 NZST — en (smoke)
- dataset: /home/jay/p4p/datasets/en_qwen3_synth (116 clips)
- base checkpoint: /home/jay/p4p/piper_training/training_output_en_qwen3_synth/lightning_logs/version_1/checkpoints/epoch=6340-step=26.ckpt (latest fine-tuned checkpoint)
- espeak voice: en-GB-x-rp
- hyperparams: batch=8 accum=1 lr=0.00002 lr_d=0.00001 max_epochs=6342 sample_rate=22050
- output dir: /home/jay/p4p/piper_training/training_output_en_qwen3_synth
- outcome: success — final ckpt: /home/jay/p4p/piper_training/training_output_en_qwen3_synth/lightning_logs/version_2/checkpoints/epoch=6341-step=52.ckpt
