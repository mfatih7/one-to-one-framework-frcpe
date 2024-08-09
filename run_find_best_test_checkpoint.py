from config import get_config

import checkpoint

config = get_config()

experiment = config.first_experiment

config.update_output_folder(experiment)

checkpoint.find_best_test_checkpoint( config, ref_angles = [5, 10, 20] )