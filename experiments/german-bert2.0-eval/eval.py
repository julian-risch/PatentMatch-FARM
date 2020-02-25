# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Downstream runner for all experiments in specified config files."""

from pathlib import Path
from experiment import run_experiment, load_experiments
from convert_bert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch as convert_tf_to_pt
import os
from shutil import copyfile
import torch

CONFIG_FILES = {
    "germEval18Fine": Path("germEval18Fine_config.json"),
    "germEval18Coarse": Path("germEval18Coarse_config.json"),
    "germEval14": Path("germEval14_config.json")
}
bert_config_file = Path("../../saved_models/german_bert_v2_wwm/bert_config.json")
checkpoints_folder = Path("../../saved_models/german_bert_v2_wwm")
vocab_file = Path("../../saved_models/german_bert_v2_wwm/vocab.txt")
mlflow_url = "https://public-mlflow.deepset.ai/"
mlflow_experiment = "Whole Word Masking"

def convert_checkpoints(dir):
    tf_checkpoints_names = fetch_tf_checkpoints(dir)
    tf_checkpoints = [dir / tfcn for tfcn in tf_checkpoints_names]
    hf_checkpoints = []
    for tfc in tf_checkpoints:
        dump_dir_name = "pt_bert_" + str(tfc).split("-")[1]
        dump_dir = checkpoints_folder / dump_dir_name
        if not os.path.isdir(dump_dir):
            os.mkdir(dump_dir)
        hf_checkpoints.append(dump_dir)
        convert_tf_to_pt(tfc, bert_config_file, dump_dir / "pytorch_model.bin")
        copyfile(bert_config_file, dump_dir / "config.json")
        copyfile(vocab_file, dump_dir / "vocab.txt")

def fetch_tf_checkpoints(dir):
    files = os.listdir(dir)
    files = [f for f in files if "model.ckpt-" in f]
    checkpoints = set(".".join(f.split(".")[:2]) for f in files)
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]), reverse=True)
    return checkpoints

def fetch_pt_checkpoints(dir):
    files = os.listdir(dir)
    files = sorted([dir / f for f in files if "pt_" in f], reverse=True)
    return files

def main():
    # NOTE: This only needs to be run once
    convert_checkpoints(checkpoints_folder)

    checkpoints = fetch_pt_checkpoints(checkpoints_folder)
    print(f"Performing evaluation on these checkpoints: {checkpoints}")
    print(f"Performing evaluation using these experiments: {CONFIG_FILES}")
    for checkpoint in checkpoints:
        for i, (conf_name, conf_file) in enumerate(CONFIG_FILES.items()):
            experiments = load_experiments(conf_file)
            steps = str(checkpoint).split("_")[-1]
            for j, experiment in enumerate(experiments):
                mlflow_run_name = f"{conf_name}_step{steps}_{j}"
                experiment.logging.mlflow_url = mlflow_url
                experiment.logging.mlflow_experiment = mlflow_experiment
                experiment.logging.mlflow_run_name = mlflow_run_name
                experiment.parameter.model = checkpoint
                experiment.general.output_dir = str(checkpoint).split("/")[:-1]
                run_experiment(experiment)
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
