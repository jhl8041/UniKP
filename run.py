import gc
import logging
import math
import pickle
import random
import re
import time
import warnings
from datetime import datetime
from enum import Enum
from typing import Union, Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from transformers import T5EncoderModel, T5Tokenizer

from executable.build_vocab import WordVocab
from executable.pretrain_trfm import TrfmSeq2seq
from utils import split

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Feature(Enum):
    BASE = (1, 'base', 'input/input_base.xlsx', 'trained_dataset/UniKP for kcat.pkl')
    PH = (2, 'pH', 'input/input_pH.xlsx', 'trained_dataset/features_636_pH_PreKcat.pkl')
    TEMPERATURE = (3, 'temperature', 'input/input_temperature.xlsx', 'trained_dataset/features_572_degree_PreKcat.pkl')

    @classmethod
    def get_by_value(cls, value: Any) -> Union['Feature', None]:
        for feature in cls:
            if value == feature.value[0]:
                return feature
        return None


class Unit(Enum):
    KCAT = (1, 'kcat', 'trained_dataset/UniKP for kcat.pkl')
    KM = (2, 'km', 'trained_dataset/UniKP for km.pkl')
    KCAT_KM = (3, 'kcat/km', 'trained_dataset/UniKP for kcat_Km.pkl')

    @classmethod
    def get_by_value(cls, value: Any) -> Union['Unit', None]:
        for unit in cls:
            if value == unit.value[0]:
                return unit
        return None


def smiles_to_vec(Smiles):
    logger.info(f"[UniKP] Start converting SMILE to vector")
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3

    vocab = WordVocab.load_vocab('trained_dataset/vocab.pkl')

    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm) > 218:
            logger.warning('SMILES is too long ({:d})'.format(len(sm)))
            sm = sm[:109]+sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1]*len(ids)
        padding = [pad_index]*(seq_len - len(ids))
        ids.extend(padding), seg.extend(padding)
        return ids, seg

    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a,b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)

    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load('trained_dataset/trfm_12_23000.pkl', map_location='cpu'))
    trfm.eval()
    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid))

    logger.info(f"[UniKP] Completed converting SMILE to vector")
    return X


def enzyme_seq_to_vec(Sequence):
    logger.info(f"[UniKP] Start converting enzyme sequence to vector")
    for i in range(len(Sequence)):
        if len(Sequence[i]) > 1000:
            logger.warning(f"[UniKP] Enzyme sequence too long for row {i+1}. length: ({len(Sequence[i])}). Truncating to 1000 characters")
            Sequence[i] = Sequence[i][:500] + Sequence[i][-500:]
    sequences_Example = []

    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        sequences_Example.append(zj)

    tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50")
    gc.collect()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    features = []

    start_time = time.time()

    for i in range(len(sequences_Example)):
        sequences_Example_i = sequences_Example[i]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)

        # Calculate elapsed time and estimate time left
        elapsed_time = time.time() - start_time
        avg_time_per_item = elapsed_time / (i + 1)
        remaining_time = format_time(avg_time_per_item * (len(sequences_Example) - (i + 1)))

        logger.info(f'[UniKP] Enzyme_seq to vector {i+1}/{len(sequences_Example)} ({(i+1)/len(sequences_Example) * 100}%). Estimated Time Left: {remaining_time}')

    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)

    for i in range(len(features)):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])

    logger.info(f"[UniKP] Completed converting enzyme sequence to vector")
    return features_normalize


def predict_base(smile_vect_list, seq_vec_list, model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    fused_vector = np.concatenate((smile_vect_list, seq_vec_list), axis=1)
    return model.predict(fused_vector)


def predict_with_environment_factor(enzyme_smile_dataset, env_dataset, kcat_dataset, enzyme_smile_input, env_input, random_seed=43):
    # 1. Prepare dataset, train 80% and validation 20%
    random.seed(random_seed)
    indices = list(range(len(enzyme_smile_dataset)))
    train_index = random.sample(indices, int(len(enzyme_smile_dataset) * 0.8))
    validation_index = [i for i in indices if i not in train_index]

    train_index = np.array(train_index)
    validation_index = np.array(validation_index)

    # 2. Generate first model (Environment factor model [pH, temperature])
    input_with_train_index = np.concatenate((enzyme_smile_dataset[train_index], env_dataset[train_index]), axis=1)
    first_model = ExtraTreesRegressor()
    first_model.fit(input_with_train_index, kcat_dataset[train_index])

    # 3. Generate second model (Fuse model)
    # Get prediction by base model
    with open("trained_dataset/UniKP for kcat.pkl", "rb") as f:
        model_base = pickle.load(f)
    base_model_prediction = model_base.predict(enzyme_smile_dataset[validation_index]).reshape([len(validation_index), 1])

    # Get prediction by first model
    input_with_validation_index = np.concatenate((enzyme_smile_dataset[validation_index], env_dataset[validation_index]), axis=1)
    first_model_prediction = first_model.predict(input_with_validation_index).reshape([len(validation_index), 1])

    # Fuse predictions to get second model
    kcat_fused = np.concatenate((base_model_prediction, first_model_prediction), axis=1)
    second_model = LinearRegression()
    second_model.fit(kcat_fused, kcat_dataset[validation_index])

    # for each enzyme_smile_input, get prediction
    base_model_kcat_result = []
    final_kcat_result = []

    start_time = time.time()

    for i in range(len(enzyme_smile_input)):
        enzyme_smile_row = [enzyme_smile_input[i]]
        target_env = env_input[i]

        target_pH_input = np.full([1, 1], target_env)  # pH or temperature
        my_input = np.concatenate((enzyme_smile_row, target_pH_input), axis=1)
        first_model_kcat_prediction = first_model.predict(my_input).reshape([1, 1])
        base_model_kcat_prediction = model_base.predict(enzyme_smile_row).reshape([1, 1])
        fused_kcat_prediction = np.concatenate((base_model_kcat_prediction, first_model_kcat_prediction), axis=1)

        base_model_kcat_result.extend(np.array(base_model_kcat_prediction).reshape([1]))
        final_kcat_result.extend(second_model.predict(fused_kcat_prediction).reshape([1]))

        # Calculate elapsed time and estimate time left
        elapsed_time = time.time() - start_time
        avg_time_per_item = elapsed_time / (i + 1)
        remaining_time = format_time(avg_time_per_item * (len(enzyme_smile_input) - (i + 1)))

        logger.info(f"[UniKP] Completed prediction for row {i+1}/{len(enzyme_smile_input)} ({(i+1)/len(enzyme_smile_input) * 100}%). Estimated Time Left: {remaining_time}")

    return base_model_kcat_result, final_kcat_result


def get_feature_choice():
    print("Choose prediction type:")
    for feature in Feature:
        print(f"{feature.value[0]}) {feature.value[1]}")
    return Feature.get_by_value(int(input("Enter your choice: ")))


def get_unit_choice(feature):
    print("Choose the output unit:")
    if feature == Feature.BASE:
        for unit in Unit:
            print(f"{unit.value[0]}) {unit.value[1]}")
        input_unit = int(input("Enter your choice: "))
    else:
        print(f"{Unit.KCAT.value[0]}) {Unit.KCAT.value[1]}")
        input_unit = int(input("Enter your choice: "))
        if input_unit != Unit.KCAT.value[0]:
            raise ValueError("Invalid unit choice. Only kcat unit is supported for pH and temperature prediction.")

    return Unit.get_by_value(input_unit)


def seq2smile(seq):
    mol = Chem.MolFromSequence(seq)
    return Chem.MolToSmiles(mol)


def transform_log(kcat):
    for i in range(len(kcat)):
        kcat[i] = math.log(kcat[i], 10)
    return kcat


def reverse_transform_log(kcat):
    for i in range(len(kcat)):
        kcat[i] = math.pow(10, kcat[i])
    return kcat


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.2f}s"


def main():
    # Get input from user
    feature = get_feature_choice()
    unit = get_unit_choice(feature)

    base_model_path = unit.value[2]

    logger.info(f"[UniKP] ======== Start prediction for feature: {feature.name}, unit: {unit.value[1]} ========")

    # Load input data
    logger.info(f"[UniKP] Input file: {feature.value[2]}")
    data = pd.read_excel(feature.value[2])

    enzyme_seq_list = data['enzyme_seq'].tolist()
    substrate_seq_list = data['substrate_seq'].tolist()
    if len(enzyme_seq_list) != len(substrate_seq_list):
        raise ValueError(f"List size doesn't match. enzyme_seq_list size: {len(enzyme_seq_list)}, substrate_seq_list size: {len(substrate_seq_list)}")

    smile_list = [seq2smile(seq) for seq in substrate_seq_list]

    data_frame_map = {
        'enzyme_seq': enzyme_seq_list,
        'substrate_seq': substrate_seq_list,
        'SMILE': smile_list
    }

    # Convert enzyme_seq and SMILE to vector
    enzyme_seq_vec_list = enzyme_seq_to_vec(enzyme_seq_list)
    smiles_vec_list = smiles_to_vec(smile_list)

    logger.info(f"[UniKP] Start prediction process...")
    if feature == Feature.BASE:
        predictions = predict_base(smiles_vec_list, enzyme_seq_vec_list, base_model_path)
        data_frame_map[unit.value[1]] = reverse_transform_log(predictions)

    elif feature == Feature.PH:
        # Load pH dataset
        database = np.array(pd.read_excel('datasets/Generated_pH_unified_smiles_636.xlsx')).T
        kcat_dataset = database[4]
        kcat_dataset_log = transform_log(kcat_dataset)
        pH_dataset = np.array(database[5]).reshape([len(kcat_dataset), 1])

        with open("trained_dataset/features_636_pH_PreKcat.pkl", "rb") as f:
            enzyme_smile_dataset = pickle.load(f)

        pH_input = data['pH'].tolist()
        if len(enzyme_seq_list) != len(pH_input):
            raise ValueError(f"List size doesn't match. enzyme_seq_list size: {len(enzyme_seq_list)}, pH_input size: {len(pH_input)}")

        enzyme_smile_input = np.concatenate((smiles_vec_list, enzyme_seq_vec_list), axis=1)

        # Get prediction for pH
        base_prediction, pH_prediction = predict_with_environment_factor(enzyme_smile_dataset, pH_dataset, kcat_dataset_log, enzyme_smile_input, pH_input)

        data_frame_map['kcat_base'] = reverse_transform_log(base_prediction)
        data_frame_map['pH'] = pH_input
        data_frame_map['kcat_pH'] = reverse_transform_log(pH_prediction)

    elif feature == Feature.TEMPERATURE:
        # Load temperature dataset
        database = np.array(pd.read_excel('datasets/Generated_degree_unified_smiles_572.xlsx')).T
        kcat_dataset = database[4]
        kcat_dataset_log = transform_log(kcat_dataset)
        degree_dataset = np.array(database[5]).reshape([len(kcat_dataset), 1])

        with open("trained_dataset/features_572_degree_PreKcat.pkl", "rb") as f:
            enzyme_smile_dataset = pickle.load(f)

        degree_input = data['degree'].tolist()
        if len(enzyme_seq_list) != len(degree_input):
            raise ValueError(f"List size doesn't match. enzyme_seq_list size: {len(enzyme_seq_list)}, degree_input size: {len(degree_input)}")

        enzyme_smile_input = np.concatenate((smiles_vec_list, enzyme_seq_vec_list), axis=1)

        # Get prediction for temperature
        base_prediction, temperature_prediction = predict_with_environment_factor(enzyme_smile_dataset, degree_dataset, kcat_dataset_log, enzyme_smile_input, degree_input)

        data_frame_map['kcat_base'] = reverse_transform_log(base_prediction)
        data_frame_map['degree_input'] = degree_input
        data_frame_map['kcat_temperature'] = reverse_transform_log(temperature_prediction)

    else:
        raise ValueError("Invalid feature choice. Please choose a valid feature.")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_df = pd.DataFrame(data_frame_map)
    file_name = f'{feature.name}_{unit.name}_{timestamp}.xlsx'
    result_df.to_excel(f'result/{file_name}', index=False)

    # Validate input data
    logger.info(f"Done. Results saved to 'result/{file_name}'")


if __name__ == '__main__':
    main()
