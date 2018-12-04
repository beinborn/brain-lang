
import numpy as np
import os
import pickle
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


def add_to_collected_results( current_results, collected_results):
    for key, value in current_results.items():

        if key in collected_results.keys():
            results = collected_results[key]
            collected_results[key] = results + value
        else:
            collected_results[key] = value
    return collected_results


def update_results(current_results,collected_results):
    for key, value in current_results.items():
        i = 0
        for eval_type in ["voxelwise", "average", "sum"]:
            my_key = key + "_" + eval_type
            my_value = value[i]
            if my_key in collected_results.keys():
                results = collected_results[my_key]
                results.append(my_value)
                collected_results[my_key] = results
            else:
                collected_results[my_key] = [my_value]
            i += 1
    return collected_results

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

def save_evaluation(evaluation_file, evaluation_name, subject_id, collected_results):
    path = os.path.dirname(evaluation_file)
    os.makedirs(os.path.dirname(evaluation_file), exist_ok=True)
    voxelwise_results = {}
    with open(evaluation_file, "w") as eval_file:
        eval_file.write("Experiment:\t" + evaluation_name + "\n")
        eval_file.write("Subject:\t" + str(subject_id) + "\n")
        voxelwise_file = path + "/voxelwise_results.pickle"
        for key, values in collected_results.items():
            result_values = np.asarray(values)
            # Write the voxelwise results to a file, so that we can also check results just for the best voxels.
            # Note: when we do no voxel selection, we still remove the voxels that are constant in the training data.
            # This might have the effect that we have a slightly different number of voxels in each fold.
            # This can affect the calculation of the sum and the calculation of the average result per voxel.
            if result_values.ndim == 2:
                average = np.mean(result_values, axis=0)
                voxelwise_results[key] = result_values

                top5 = sorted(average)[-20:]
                eval_file.write(str(key) + "_top20\t"  + str(top5.reverse()))
                with open(voxelwise_file, "wb") as handle:
                    pickle.dump(voxelwise_results, handle)

            else:
                average = np.mean(result_values)
                eval_file.write(str(key) + "\t" + str(average) + "\t" + str(values))
                print(str(key) + "\t" + str(average) + "\t" + str(values))
            eval_file.write("\n")