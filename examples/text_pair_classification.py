# fmt: off
import logging
from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.data_silo import StreamingDataSilo
from farm.data_handler.processor import RegressionProcessor, TextPairClassificationProcessor
from farm.experiment import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import RegressionHead, TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
from farm.train import Trainer, EarlyStopping
import pandas as pd
import math
from datetime import datetime

training_filename="patentmatch_train_balanced.tsv"
test_filename="patentmatch_test_balanced.tsv"


def text_pair_classification():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_text_pair_classification")

    ##########################
    ########## Settings ######
    ##########################
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    n_epochs = 2
    batch_size = 32
    evaluate_every = 500
    lang_model = "bert-base-cased"
    label_list = ["0", "1"]

    n_batches,class_weights=calc_n_batches_and_classweights(batch_size)

    print("calculated n_batches")
    print(n_batches)

    print("calculated classweights")
    print(class_weights)

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False)


    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    #    We do not have a sample dataset for regression yet, add your own dataset to run the example
    processor = TextPairClassificationProcessor(tokenizer=tokenizer,
                                                label_list=label_list,
                                                metric = "acc",
                                                label_column_name="label",
                                                max_seq_len=64,
                                                train_filename=training_filename,
                                                test_filename=test_filename,
                                                dev_filename=test_filename,
                                                #dev_split = 0.5,
                                                data_dir=Path("../data"),
                                                tasks={"text_classification"},
                                                delimiter="\t")

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = StreamingDataSilo(
        processor=processor,
        batch_size=batch_size)


    #Alte Version vor StreamingDataSilo
    #data_silo = DataSilo(
    #    processor=processor,
    #    batch_size=batch_size, max_processes=4)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.load(lang_model)
    # b) and a prediction head on top that is suited for our task
    prediction_head = TextClassificationHead(num_labels=len(label_list),
                                             class_weights=class_weights # Reihenfolge: Element i entspricht Klasse i (sklearn Methode sklearn.utils.class_weight.compute_class_weights implementiert bei farm
                                             )

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence_continuous"],
        device=device)

    # 5. Create an optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=5e-6,
        device=device,
        n_batches=n_batches,
        n_epochs=n_epochs)


    now = datetime.now() # current date and time

    # An early stopping instance can be used to save the model that performs best on the dev set
    # according to some metric and stop training when no improvement is happening for some iterations.
    earlystopping = EarlyStopping(
        #metric="f1_weighted", mode="max",  # use f1_macro from the dev evaluator of the trainer
        metric="loss", mode="min",   # use loss from the dev evaluator of the trainer
        save_dir=Path("saved_models/earlystopping/"+ now.strftime("%m%d%Y%H%M%S")),  # where to save the best model
        patience=8    # number of evaluations to wait for improvement before terminating the training
    )

    # 6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device,
        early_stopping=earlystopping)

    # 7. Let it grow
    trainer.train()

    # 8. Hooray! You have a model. Store it:
    save_dir = Path("saved_models/savingModel/"+ now.strftime("%m%d%Y%H%M%S"))
    model.save(save_dir)
    processor.save(save_dir)

    # 9. Load it & harvest your fruits (Inference)
    #    Add your own text adapted to the dataset you provide
    basic_texts = [
        {"text": "<claim-text>The method of claim 10, wherein the indium metal layer is 10 nm to 100 µm thick.</claim-text>", "text_b": "<p id=""p0001"" num=""0001"">The present invention is directed to metal plating compositions and methods. More specifically, the present invention is directed to metal plating compositions and methods which provide improved leveling and throwing power.</p <p id=""p0039"" num=""0039"">One or more conventional surfactants may be used. Typically, surfactants include, but are not limited to, nonionic surfactants such as alkyl phenoxy polyethoxyethanols. Other suitable surfactants containing multiple oxyethylene groups also may be used. Such surfactants include compounds of polyoxyethylene polymers having from as many as 20 to 150 repeating units. Such compounds also may perform as suppressors. Also included in the class of polymers are both block and random copolymers of polyoxyethylene (EO) and polyoxypropylene (PO). Surfactants may be added in conventional amounts, such as from 0.05 g/L to 20 g/L or such as from 0.5 g/L to 5 g/L.</p <p id=""p0040"" num=""0040"">Conventional levelers include, but are not limited to, one or more of alkylated polyalkyleneimines and organic sulfo sulfonates. Examples of such compounds include, 4-mercaptopyridine, 2-mercaptothiazoline, ethylene thiourea, thiourea, 1-(2-hydroxyethyl)-2-imidazolidinethion (HIT) and alkylated polyalkyleneimines. Such levelers are included in conventional amounts. Typically, such levelers are included in amounts of 1ppb to 1 g/L, or such as from 10ppb to 500ppm.</p <p id=""p0042"" num=""0042"">Alkali metal salts which may be included in the plating compositions include, but are not limited to, sodium and potassium salts of halogens, such as chloride, fluoride and bromide. Typically chloride is used. Such alkali metal salts are used in conventional amounts.</p <p id=""p0053"" num=""0053"">The metal plating compositions may be used to plate a metal or metal alloy on a substrate by any method known in the art and literature. Typically, the metal or metal alloy is electroplated using conventional electroplating processes with conventional apparatus. A soluble or insoluble anode may be used with the electroplating compositions.</p <p id=""p0022"" num=""0022"">One or more sources of metal ions are included in metal plating compositions to plate metals. The one or more sources of metal ions provide metal ions which include, but are not limited to, copper, tin, nickel, gold, silver, palladium, platinum and indium. Alloys include, but are not limited to, binary and ternary alloys of the foregoing metals. Typically, metals chosen from copper, tin, nickel, gold, silver or indium are plated with the metal plating compositions. More typically, metals chosen from copper, tin, silver or indium are plated. Most typically, copper is plated.</p <p id=""p0030"" num=""0030"">Indium salts which may be used include, but are not limited to, one or more of indium salts of alkane sulfonic acids and aromatic sulfonic acids, such as methanesulfonic acid, ethanesulfonic acid, butane sulfonic acid, benzenesulfonic acid and toluenesulfonic acid, salts of sulfamic acid, sulfate salts, chloride and bromide salts of indium, nitrate salts, hydroxide salts, indium oxides, fluoroborate salts, indium salts of carboxylic acids, such as citric acid, acetoacetic acid, glyoxylic acid, pyruvic acid, glycolic acid, malonic acid, hydroxamic acid, iminodiacetic acid, salicylic acid, glyceric acid, succinic acid, malic acid, tartaric acid, hydroxybutyric acid, indium salts of amino acids, such as arginine, aspartic acid, asparagine, glutamic acid, glycine, glutamine, leucine, lysine, threonine, isoleucine, and valine.</p"},
        {"text": "<claim-text>A toner comprising: <claim-text>toner base particles; and</claim-text> <claim-text>an external additive,</claim-text> <claim-text>the toner base particles each comprising a binder resin and a colorant,</claim-text> <claim-text>wherein the external additive comprises coalesced particles,</claim-text> <claim-text>wherein the coalesced particles are each a non-spherical secondary particle in which primary particles are coalesced together, and</claim-text> <claim-text>wherein an index of a particle size distribution of the coalesced particles is expressed by the following Formula (1): <maths id=""math0004"" num=""(formula (1)""><math display=""block""><mfrac><msub><mi>Db</mi><mn>50</mn></msub><msub><mi>Db</mi><mn>10</mn></msub></mfrac><mo>≦</mo><mn>1.20</mn></math><img id=""ib0008"" file=""imgb0008.tif"" wi=""93"" he=""21"" img-content=""math"" img-format=""tif""/></maths><br/> where, in a distribution diagram in which particle diameters in nm of the coalesced particles are on a horizontal axis and cumulative percentages in % by number of the coalesced particles are on a vertical axis and in which the coalesced particles are accumulated from the coalesced particles having smaller particle diameters to the coalesced particles having larger particle diameters, Db<sub>50</sub> denotes a particle diameter of the coalesced particle at which the cumulative percentage is 50% by number, and Db<sub>10</sub> denotes a particle diameter of the coalesced particle at which the cumulative percentage is 10% by number.</claim-text></claim-text>", "text_b": "<p id=""p0177"" num=""0177"">For a similar reason, it is preferred that the electroconductive fine powder has a volume-average particle size of 0.5 - 5 µm, more preferably 0.8 - 5 µm, further preferably 1.1 - 5 µm and has a particle size distribution such that particles of 0.5 µm or smaller occupy at most 70 % by volume and particles of 5.0 µm or larger occupy at most 5 % by number.</p <p id=""p0189"" num=""0189"">The volume-average particle size and particle size distribution of the electroconductive fine powder described herein are based on values measured in the following manner. A laser diffraction-type particle size distribution measurement apparatus (""Model LS-230"", available from Coulter Electronics Inc.) is equipped with a liquid module, and the measurement is performed in a particle size range of 0.04 - 2000 µm to obtain a volume-basis particle size distribution. For the measurement, a minor amount of surfactant is added to 10 cc of pure water and 10 mg of a sample electroconductive fine powder is added thereto, followed by 10 min. of dispersion by means of an ultrasonic disperser (ultrasonic homogenizer) to obtain a sample dispersion liquid, which is subjected to a single time of measurement for 90 sec.</p <p id=""p0191"" num=""0191"">In the case where the electroconductive fine powder is composed of agglomerate particles, the particle size of the electroconductive fine powder is determined as the particle size of the agglomerate. The electroconductive fine powder in the form of agglomerated secondary particles can be used as well as that in the form of primary particles. Regardless of its agglomerated form, the electroconductive fine powder can exhibit its desired function of charging promotion by presence in the form of the agglomerate in the charging section at the contact position<!-- EPO <DP n=""85""> --> between the charging member and the image-bearing member or in a region in proximity thereto.</p"},
    ]

    model = Inferencer.load(save_dir)
    result = model.inference_from_dicts(dicts=basic_texts)

    print(result)

def calc_n_batches_and_classweights(batch_size):
    df = pd.read_csv(Path("../data/"+training_filename), sep='\t', header=0)
    class_0=(df['label'] == 0).sum()
    class_1=(df['label'] == 1).sum()
    samples=df.shape[0]
    n_batches=math.ceil(samples/batch_size)

    return n_batches, [class_1/samples,class_0/samples]



if __name__ == "__main__":
    text_pair_classification()

# fmt: on
