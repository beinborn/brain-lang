"""Pipeline for predicting brain activations.
"""


from voxel_preprocessing.select_voxels import *
import numpy as np
import logging
import random

logging.basicConfig(level=logging.INFO)


class Pipeline(object):
    def __init__(self, brain_data_reader, stimuli_encoder, mapper, pipeline_name, save_dir="processed_data/"):
        self.brain_data_reader = brain_data_reader
        self.stimuli_encoder = stimuli_encoder
        self.mapper = mapper
        self.subject_ids = None
        self.pipeline_name = pipeline_name


    def process(self):
        raise NotImplemented

    # This method is in here, because it needs to access the mapper.
    # Voxel selection settings are: "none", "on_train_ev", "on_train_r", "random"
    def select_interesting_voxels(self, voxel_selection_mode, stimuli, scans, experiment_name,
                                  number_of_selected_voxels=500):
        # Default: only ignore constant voxels
        all_voxels = len(scans[0])
        varied_voxels = select_varied_voxels(scans)

        selected_voxels = varied_voxels
        if voxel_selection_mode == "none":
            logging.info("Not applying any selection. Just excluding the constant voxels. Number of kept voxels: " +str(len(varied_voxels)))
            return varied_voxels

        if voxel_selection_mode == "roi":
            return select_voxels_by_roi(scans)

        if voxel_selection_mode == "random":
            logging.info("Selecting voxels RANDOMLY")
            random.seed = 5
            random.shuffle(selected_voxels)
            selected_voxels = selected_voxels[:number_of_selected_voxels]
            logging.info("Voxel selection completed. Selected: " + str(len(selected_voxels)))
            return selected_voxels

        # Data-driven selection on the training data
        if voxel_selection_mode.startswith("on_train"):

            # Train the mapper
            self.mapper.train(stimuli, scans)
            logging.info("Voxel evaluation model is trained.")

            # Get predictions
            train_predictions = self.mapper.map(inputs=stimuli)["predictions"]

            # Select the voxels
            if self.voxel_selection.endswith("r"):
                logging.info("Evaluate voxels BY R")
                selected_voxels = select_voxels_by_r(train_predictions, scans)
            else:
                logging.info("Evaluate voxels BY EV")
                selected_voxels = select_voxels_by_variance(train_predictions, scans)
            logging.info("Voxels are evaluated")

            # Take intersection of predictive and non-constant voxels
            # This can take long.
            # Would probably be faster to subtract the constant voxels.
            if (len(varied_voxels) < all_voxels):

                selected_voxels= [voxel for voxel in selected_voxels if voxel in set(varied_voxels)]

            topn_voxels = selected_voxels[-number_of_selected_voxels:]
            logging.info("Voxel selection completed. Selected: " + str(len(topn_voxels)))

            voxel_file = self.save_dir + "/" + self.pipeline_name + "/" + experiment_name + "selected_voxels.txt"
            with open(voxel_file, "a") as savefile:
                savefile.write(str(topn_voxels))
                savefile.write("\n")
            return topn_voxels
        else:
            raise ValueError("Choose a voxel selection mode. You can use 'none'.")


# We used this method to determine the most predictive voxels (=metric score higher than threshold) for the brain plot.
# We don't think it is a very reasonable approach to select voxels on the test set, but it might be interesting for analyses.
    def get_predictive_voxels(self, metric_fn, train_stimuli, train_targets, test_stimuli, test_targets, threshold,  experiment_name):
        # Train model
        self.mapper.train(train_stimuli, train_targets)

        # Predict
        predictions = self.mapper.map(inputs=test_stimuli)["predictions"]

        # Get results
        score_per_voxel = metric_fn(test_targets, predictions, multioutput="raw_values")



        # Get voxels above threshold
        predictive_voxels =np.argwhere(np.asarray(score_per_voxel)>threshold).tolist()

        # Remove constant voxels
        all_voxels = len(train_targets[0])
        varied_voxels = set(select_varied_voxels(train_targets))
        print(all_voxels)
        print(len(predictive_voxels))
        if len(varied_voxels) < all_voxels:
            predictive_voxels = [v for v in predictive_voxels if v[0] in varied_voxels]
            indices = predictive_voxels
        else:
            indices = [x[0] for x in predictive_voxels]


        voxel_file = self.save_dir + "/" + self.pipeline_name + "/" + experiment_name + "predictive_voxels.txt"
        with open(voxel_file, "w") as savefile:
            savefile.write("Explained variance scores: ")
            savefile.write(str([str(score) for score in score_per_voxel]))
            savefile.write("\n")
            savefile.write("Voxels with explained variance above threshold " + str(threshold) +":  ")
            savefile.write(str(indices))
            savefile.write("\n")

        return indices



