"""Dataset loading and preprocessing."""

import os.path

import numpy as np
import pandas as pd
import tensorflow as tf
# Disallow TensorFlow from using GPU so that it won't interfere with JAX.
tf.config.set_visible_devices([], 'GPU')

train_csvs_dir = "/home/szaiem/rich_representation_learning/data/slurp_csvs/"
encoder_dir = "/export/corpora/representations/SLURP/encoder_outputs/"

train_csv = "/home/szaiem/rich_representation_learning/data/slurp_csvs/train_real-type=direct.csv"
train_synthetic = "/home/szaiem/rich_representation_learning/data/slurp_csvs/train_synthetic-type=direct.csv"
test_csv = "/home/szaiem/rich_representation_learning/data/slurp_csvs/test-type=direct.csv"
dev_csv = "/home/szaiem/rich_representation_learning/data/slurp_csvs/devel-type=direct.csv"


def create_label_encoders(train_csvs):
    actions_dict = {}
    scen_dict = {}
    intent_dict = {}
    intent_count = 0
    act_count = 0
    scen_count = 0
    tables = []
    for train_csv in train_csvs:
        tables.append(pd.read_csv(train_csv))
    table = pd.concat(tables)
    semantics = list(table["semantics"])
    actions = []
    scenarios = []
    intents = []
    for sem in semantics:
        scenario, action, intent = get_scenario_action(sem)
        intents.append(intent)
        actions.append(action)
        scenarios.append(scenario)
    unique_actions = set(actions)
    unique_scenarios = set(scenarios)
    unique_intents = set(intents)
    for act in unique_actions:
        actions_dict[act] = act_count
        act_count += 1
    for scen in unique_scenarios:
        scen_dict[scen] = scen_count
        scen_count += 1
    for intent in unique_intents:
        intent_dict[intent] = intent_count
        intent_count += 1

    return scen_dict, actions_dict, scen_count, act_count, intent_dict, intent_count


scen_dict, actions_dict, scen_count, act_count, intent_dict, intent_count = create_label_encoders(
    [train_csv, train_synthetic]
)
print(f" total number of scenarios : {scen_count}")
print(f" total number of actions : {act_count}")
print(f" total number of intents : {intent_count}")


def preprocess_example_tf(wav, scenario, action, intent):
    b = tf.numpy_function(
        numpy_preprocess, [wav, scenario, action, intent],
        [tf.float32, tf.int64, tf.int64, tf.int64, tf.int64]
    )
    return {
        "encoder_frames": b[0],
        "action": b[2],
        "scenario": b[1],
        "intent": b[3],
        'num_frames': b[4]
    }


def numpy_preprocess(wav, scenario, action, intent):
    encoder_frames = np.squeeze(
        np.load(os.path.join(encoder_dir,
                             wav.decode("utf-8") + ".npy"))
    )
    action_label, scenario_label, intent_label = actions_dict[
        action.decode("utf-8")], scen_dict[scenario.decode("utf-8")
                                          ], intent_dict[intent.decode("utf-8")]
    return encoder_frames, scenario_label, action_label, intent_label, encoder_frames.shape[
        1]


list_of_lengths = [100, 250, 400, 600]


def preprocess(
    dataset: tf.data.Dataset,
    is_train: bool = True,
    batch_size: int = 6,
    max_num_frames: int = 600,
) -> tf.data.Dataset:
    """Applies data preprocessing for training and evaluation."""
    # Preprocess individual examples.
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000).repeat()
    dataset = dataset.map(
        preprocess_example_tf, num_parallel_calls=tf.data.AUTOTUNE
    )
    # Shuffle and repeat data for training.
    # Pad and batch examples.
    dataset = dataset.padded_batch(
        batch_size,
        {
            'encoder_frames': [max_num_frames, None],
            'action': [],
            'scenario': [],
            "intent": [],
            'num_frames': [],
        },
    )
    return dataset


def create_dict(train_csvs, test=False):
    tables = []
    for train_csv in train_csvs:
        tables.append(pd.read_csv(train_csv))
    table = pd.concat(tables)
    if test:
        table = table[0:13074]
    semantics = table["semantics"]
    file_ids = [x.split("/")[-1] for x in list(table["wav"])]
    actions = []
    scenarios = []
    intents = []
    for sem in semantics:
        scenario, action, intent = get_scenario_action(sem)
        actions.append(action)
        scenarios.append(scenario)
        intents.append(intent)
    return file_ids, scenarios, actions, intents


train_all = create_dict([train_csv, train_synthetic])
test_all = create_dict([test_csv], test=True)
test_all = create_dict([train_csv], test=False)
#test_all = create_dict([train_csv, train_synthetic])
dev_all = create_dict([dev_csv])
train_dataset = tf.data.Dataset.from_tensor_slices(train_all)
test_dataset = tf.data.Dataset.from_tensor_slices(test_all)
dev_dataset = tf.data.Dataset.from_tensor_slices(dev_all)

TEST_BATCH_SPLIT = 3
# A single test batch.
DEV_BATCH = next(
    dev_dataset.take(TEST_BATCH_SPLIT).apply(
        functools.partial(preprocess, batch_size=TEST_BATCH_SPLIT)
    ).as_numpy_iterator()
)

TEST_BATCHES = (
    test_dataset.apply(functools.partial(preprocess)
                      ).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
)

# An iterator of training batches.
TRAIN_BATCHES = (
    train_dataset.apply(functools.partial(preprocess)
                       ).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
)
