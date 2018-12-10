
import numpy as np
import os
import pickle

# These methods are used to collect and save the evaluations per fold.

# Collect results in a list
def append_to_collected_results( current_results, collected_results):
    for key, value in current_results.items():

        if key in collected_results.keys():
            results = collected_results[key]
            if (len(results)>1):
                results.append([results])
            else:
                results.append(value)
            collected_results[key] = results
        else:
            collected_results[key] = [value]
    return collected_results


# Add results to previous results
def add_to_collected_results( current_results, collected_results):
    for key, value in current_results.items():

        if key in collected_results.keys():
            results = collected_results[key]
            collected_results[key] = results + value
        else:
            collected_results[key] = value
    return collected_results


# Update a dictionary of results.
# This method is very specific, might have to be updated, in case you change the evaluation setup.
def update_results(current_results,collected_results):
    for key, value in current_results.items():
        i = 0
        for voxel_number in ["all", "top"]:
            for eval_type in ["voxelwise", "average", "sum"]:
                my_key = voxel_number + "_" +key + "_" + eval_type
                my_value = value[i]
                if my_key in collected_results.keys():
                    results = collected_results[my_key]
                    results.append(my_value)
                    collected_results[my_key] = results
                else:
                    collected_results[my_key] = [my_value]
                i += 1
    return collected_results

# Some evaluation metrics do not work with constant rows.
# Careful: this method messes up the voxel ids.
# In case you need them, use voxel_preprocessing.select_voxels.select_varied_voxels instead.
def delete_constant_rows(predictions, targets):
    index = 0
    num_deleted = 0
    for i in targets:
        if (np.var(i) == 0):
            targets = np.delete(targets, index, 0)
            predictions = np.delete(predictions, index, 0)
            num_deleted += 1
        index += 1

    return predictions, targets


# Save results for the pairwise evaluation procedure.
def save_pairwise_evaluation(evaluation_file, evaluation_name, subject_id, collected_matches,
                             number_of_pairs):
    path = os.path.dirname(evaluation_file)
    os.makedirs(os.path.dirname(evaluation_file), exist_ok=True)
    with open(evaluation_file, "w") as eval_file:
        eval_file.write("Experiment:\t" + evaluation_name + "\n")
        eval_file.write("Subject:\t" + str(subject_id) + "\n")
        for key, value in collected_matches.items():
            print("\n")
            print(key)
            print(value, number_of_pairs)
            accuracy = value / float(number_of_pairs)
            print(accuracy)
            eval_file.write(str(key) + "\t" + str(accuracy) + "\n")

# Save the evaluation for voxelwise results.
def save_evaluation(evaluation_file, evaluation_name, subject_id, collected_results):
    path = os.path.dirname(evaluation_file)
    os.makedirs(os.path.dirname(evaluation_file), exist_ok=True)
    voxelwise_results = {}
    with open(evaluation_file, "w") as eval_file:
        eval_file.write("Experiment:\t" + evaluation_name + "\n")
        eval_file.write("Subject:\t" + str(subject_id) + "\n")

        for key, values in collected_results.items():
            result_values = np.asarray(values)
            # Write the voxelwise results to an extra file.
            # Note: when we do no voxel selection, we still remove the voxels that are constant in the training data.
            # This might have the effect that we have a slightly different number of voxels in each fold.
            # This can affect the calculation of the sum and the calculation of the average result per voxel.
            if  key.endswith("voxelwise"):

                voxelwise_file = path + "/" + evaluation_name + "_" + key + "voxelwise_results.pickle"
                voxelwise_results[key] = result_values

                with open(voxelwise_file, "wb") as handle:
                    pickle.dump(voxelwise_results, handle)

            # Average the results over all folds
            else:
                average = np.mean(result_values)
                eval_file.write(str(key) + "\t" + str(average) + "\t" + str(values))
                print(str(key) + "\t" + str(average) + "\t" + str(values))
            eval_file.write("\n")