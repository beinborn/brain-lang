import pickle
import pandas as pd
import glob, os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# resultfile = "/Users/Lisa/Experiments/fmri/Continuous/HarryElmoEncoder/standard512_none_Pearson_squared_voxelwisevoxelwise_results.pickle"
# with open(resultfile, 'rb') as handle:
#     results = pickle.load(handle)
# ps = results["Pearson_squared_voxelwise"]
# print(len(ps))
# print(ps)
# i = 0
# for result in ps:
#     for value in result:
#         if np.isnan(value):
#             print(i)
#             print(result)
#         i+=1





# This method can be used to combine all evaluation files in result_dir with the same pattern and write them to outfile
# We assume, that the the metric name is in the first column and the results are in the second colum.
# All files need to have the same number of results in the same order.
def combine_results(result_dir, pattern, outfile):
    os.chdir(result_dir)
    results = pd.DataFrame([])

    for counter, file in enumerate(glob.glob(pattern)):
        data = pd.read_csv(file, sep='\t', usecols=[0, 1])
        results = pd.concat([results, data.iloc[:, 1]], axis=1, ignore_index=True)

    # Add names to beginning

    results.insert(0, "", data.iloc[:, 0])
    results.set_index( [0])
    print(results.index)
    if "cv" in pattern:
        make_cv_plot(results, outfile)
    else:
        make_pairwise_plot(results, outfile)


# Add column for average
    results = results.round(4)
    results['average'] = results.mean(numeric_only=True, axis=1).round(4)
    results['average'].iloc[0] = "Mean over all subjects"
    results.to_csv(outfile+".csv", sep="\t", header=False, index=False)

def make_cv_plot(results, outfile):
    # This is the code for the violin plot of the Kaplan data.
    # It is very cumbersome, could be better automated,
    # if I figure out how to properly read in the data with headers and index.
    # When I have time...

    r2sum = pd.to_numeric(results.iloc[2, 1:])
    evsum = pd.to_numeric(results.iloc[4, 1:])
    r2jainsum = pd.to_numeric(results.iloc[6, 1:])

    sns.set(style="whitegrid")
    fig, axs = plt.subplots(ncols=3, sharex='row', sharey="row")

    plot1 = sns.violinplot(y=r2sum, color="darkorange", ax=axs[0])
    plot1.set(xlabel='R2', ylabel='Sum')
    plot1.xaxis.set_label_position('top')
    plot2 = sns.violinplot(y=evsum, color="darkcyan", ax=axs[1])
    plot2.set(xlabel='EV', ylabel='')
    plot2.xaxis.set_label_position('top')
    plot3 = sns.violinplot(y=r2jainsum, color="darkmagenta", ax=axs[2])
    plot3.set(xlabel='R2_simple', ylabel="")
    plot3.xaxis.set_label_position('top')
    fig.add_subplot(132, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel("Sum over all voxels")
    plt.ylabel("")
    plt.savefig(outfile + "crossvalidation.png")
    plt.show()

def make_pairwise_plot(results, outfile):


    # Pairwise plot
    cosine = pd.to_numeric(results.iloc[1,1:])
    cosine_wehbe = pd.to_numeric(results.iloc[2, 1:])
    cosine_strict = pd.to_numeric(results.iloc[4, 1:])
    euclidean = pd.to_numeric(results.iloc[5, 1:])
    euclidean_wehbe = pd.to_numeric(results.iloc[6, 1:])
    euclidean_strict = pd.to_numeric(results.iloc[8, 1:])
    pearson = pd.to_numeric(results.iloc[9, 1:])
    pearson_wehbe = pd.to_numeric(results.iloc[10, 1:])
    pearson_strict = pd.to_numeric(results.iloc[12, 1:])

    print(cosine)
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(ncols=9,  sharex='row', sharey="row")

    plot1 = sns.violinplot(y=cosine,color = "darkorange" , ax=axs[0])
    plot1.set(xlabel='Cos', ylabel='Accuracy')
    plot1.xaxis.set_label_position('top')
    plot2 = sns.violinplot(y=cosine_wehbe, color="darkcyan", ax=axs[3])
    plot2.set(xlabel='Cos', ylabel = "")
    plot2.xaxis.set_label_position('top')
    plot3 = sns.violinplot(y=cosine_strict,color = "darkmagenta" , ax=axs[6])
    plot3.set(xlabel='Cos', ylabel="")
    plot3.xaxis.set_label_position('top')
    plot4 = sns.violinplot(y=euclidean,color = "darkorange" , ax=axs[1])
    plot4.set(xlabel='Eucl', ylabel="")
    plot4.xaxis.set_label_position('top')
    plot5 = sns.violinplot(y=euclidean_wehbe,color = "darkcyan" , ax=axs[4])
    plot5.set(xlabel='Eucl', ylabel="")
    plot5.xaxis.set_label_position('top')
    plot6 = sns.violinplot(y=euclidean_strict, color="darkmagenta", ax=axs[7])
    plot6.set(xlabel='Eucl', ylabel="")
    plot6.xaxis.set_label_position('top')
    plot7 = sns.violinplot(y= pearson,color = "darkorange" , ax=axs[2])
    plot7.set(xlabel='R', ylabel="")
    plot7.xaxis.set_label_position('top')
    plot8 = sns.violinplot(y=pearson_wehbe,color = "darkcyan" , ax=axs[5])
    plot8.set(xlabel='R', ylabel="")
    plot8.xaxis.set_label_position('top')
    plot9 = sns.violinplot(y=pearson_strict, color="darkmagenta", ax=axs[8])
    plot9.set(xlabel='R', ylabel="")
    plot9.xaxis.set_label_position('top')

    # add empty outside plot to have one label for three subplots
    fig.add_subplot(192, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel("Sum Match")

    fig.add_subplot(195, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel("Single Match")
    fig.add_subplot(198, frameon=False)

    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel("Strict Match")
    plt.ylabel("")
    plt.savefig(outfile + "pairwise.png")
    plt.show()


#combine_results("/Users/lisa/Experiments/fmri/single_instance/Posts_ElmoEncoder/","*_2x2.txt",#"/Users/lisa/Experiments/fmri/Analysis/PostsElmoEncoder/Posts_all_ElmoEncoder_2x2.csv")

def get_results(encoder, setting = "single_instance", vs =["on_train_ev", "none"], exp = [ "cv", "2x2"] ):
    resultdir = "/Users/lisa/Experiments/fmri/" + setting + "/"+ encoder
    savedir = "/Users/lisa/Experiments/fmri/Analysis/" + encoder
    for vs_type in vs:
        for exp_type in exp:
            print( exp_type, vs_type)
            combine_results(resultdir,"*"+vs_type + "*_"+ exp_type + "*.txt",
                            savedir +"_"+ vs_type +"_"+ exp_type )

get_results("WordsElmoEncoder")
#get_results(HarryElmoEncoder", "Continuous")
#get_results(HarryRandomEncoder",  "Continuous")
#get_results("WordsRandomEncoder")

#get_results("PostsRandomEncoder")