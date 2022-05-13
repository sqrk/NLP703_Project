import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger  

from ruamel.yaml import YAML
config_path = "../Configs/quartznet-config.yaml"

yaml = YAML(typ="safe")
with open(config_path) as f:
    params = yaml.load(f)
print(params)

params["model"]["train_ds"]["manifest_filepath"] = "../Manifests/Nemo/UASpeech-train-manifest.json"
params["model"]["validation_ds"]["manifest_filepath"] = "../Manifests/Nemo/UASpeech-test-manifest.json"

for k,v in params.items(): 
    wandb_logger.experiment.config[k]=v 

# model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(main_dir + "../Checkpoints/Quartznet_68.ckpt")

model.setup_training_data(train_data_config=params["model"]["train_ds"])
model.setup_validation_data(val_data_config=params["model"]["validation_ds"])

trainer = pl.Trainer(gpus=1, max_epochs=60) #Max epochs limited by 12h max runtime on cluster

trainer.fit(model)

wer_nums = []
wer_denoms = []

for test_batch in model.test_dataloader():
    test_batch = [x.cuda() for x in test_batch]
    targets = test_batch[2]
    targets_lengths = test_batch[3]        
    log_probs, encoded_len, greedy_predictions = model(
        input_signal=test_batch[0], input_signal_length=test_batch[1]
    )

    # Notice the model has a helper object to compute WER
    model._wer.update(greedy_predictions, targets, targets_lengths)
    _, wer_num, wer_denom = model._wer.compute()
    model._wer.reset()
    wer_nums.append(wer_num.detach().cpu().numpy())
    wer_denoms.append(wer_denom.detach().cpu().numpy())

    # Release tensors from GPU memory
    del test_batch, log_probs, targets, targets_lengths, encoded_len, greedy_predictions

print(f"WER = {sum(wer_nums)/sum(wer_denoms)}")
