from utils.input_sequence_builder import build_sequences
from utils.file_proxy import build_output_csv, dump_sets_to_csv
from utils.constraints import *
import utils.executor as executor
import config
import mdp_simulator


def create_dataset(ss_variables, max_expected, random_sampling=False):
    # Build the input dataset entries
    sets = build_sequences(ss_variables, max_expected, random_sampling)
    dump_sets_to_csv(sets)


def compute_results(ss_variables, index_to_run, total_to_train, constraints, path_to_dataset):
    result = executor.run(ss_variables, index_to_run, total_to_train, constraints, path_to_dataset)
    build_output_csv(ss_variables, result['input_data'], result['output_data'], f"{index_to_run}-{total_to_train}.csv")


if __name__ == "__main__":
    mdp_simulator.config.FOLDER_NAME = "./RescueRobot"
    mdp_simulator.config.DEBUG_LEVEL = enums.LogTypes.ERROR

    if config.MAX_SAMPLES is not None:
        print(f"Creating {config.MAX_SAMPLES} samples")
        create_dataset(config.SS_VARIABLES, config.MAX_SAMPLES, random_sampling=True)
    if config.INDEX_TO_RUN is not None and config.TOTAL_TO_RUN is not None:
        print(f"Running {config.INDEX_TO_RUN} out of {config.TOTAL_TO_RUN}")
        compute_results(config.SS_VARIABLES, config.INDEX_TO_RUN, config.TOTAL_TO_RUN, config.CONSTRAINTS, config.PATH_TO_DATASET)
