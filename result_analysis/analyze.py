import pickle
import pandas as pd
import glob, os
import seaborn as sns
import matplotlib.pyplot as plt

# We use pandas and seaborn for analysis and plotting, but the experimental code should also work without it.

# This method can be used to combine all evaluation files that have been obtained by the same encoder.
# It writes out results per subjects and averages over all subjects.
# It also produces some nice violin plots for a first impression.

def get_results(encoder, setting = "single_instance", vs =["on_train_ev", "none"], exp = [ "cv", "2x2"] ):
    resultdir = "/Users/lisa/Experiments/fmri/" + setting + "/"+ encoder +"/"
    savedir = "/Users/lisa/Experiments/fmri/Analysis/" + encoder +"/"
    for vs_type in vs:
        for exp_type in exp:
            print( exp_type, vs_type)
            combine_results(resultdir,"*"+vs_type + "*_"+ exp_type + "*.txt",
                            savedir + vs_type +"_"+ exp_type )



# This method combines all evaluation files in result_dir with the same pattern and writes them to outfile.
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

# This is the code for the violin plot of the Kaplan data for the crossvalidation.
# Results is a pandas dataframe.
# It is very cumbersome, could be better automated,
# if I figure out how to properly read in the data with headers and index.
# When I have time...
def make_cv_plot(results, outfile):
    print(results)
    # Read values we are interested in.

    r2sum = pd.to_numeric(results.iloc[2, 1:])
    evsum = pd.to_numeric(results.iloc[6, 1:])
    r2jainsum = pd.to_numeric(results.iloc[10, 1:])

    #For top 500 on test
    # r2sum = pd.to_numeric(results.iloc[4, 1:])
    # evsum = pd.to_numeric(results.iloc[8, 1:])
    # r2jainsum = pd.to_numeric(results.iloc[12, 1:])
    #Set up subplots for each value
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(ncols=3, sharex='row', sharey="row")
    plot1 = sns.violinplot(y=r2sum, color="darkorange", ax=axs[0])
    plot1.set(xlabel='$R^2$', ylabel='')
    plot1.xaxis.set_label_position('top')
    plot1.set_ylim(-250, 400)
    plot2 = sns.violinplot(y=evsum, color="darkcyan", ax=axs[1])
    plot2.set(xlabel='EV', ylabel='')
    plot2.set_ylim(-250,400)
    plot2.xaxis.set_label_position('top')
    plot3 = sns.violinplot(y=r2jainsum, color="darkmagenta", ax=axs[2])
    plot3.set(xlabel='$r^2$ simple', ylabel="" )
    plot2.set_ylim(-250, 400)
    plot3.xaxis.set_label_position('top')

    # Add empty outside plot to have one label for the subplots
    fig.add_subplot(132, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel("Sum over all voxels")
    plt.ylabel("")
    plt.savefig(outfile + "crossvalidation.png")
    plt.show()

def make_pairwise_plot(results, outfile):

    # Get results
    cosine = pd.to_numeric(results.iloc[1,1:])
    cosine_wehbe = pd.to_numeric(results.iloc[2, 1:])
    cosine_strict = pd.to_numeric(results.iloc[4, 1:])
    euclidean = pd.to_numeric(results.iloc[5, 1:])
    euclidean_wehbe = pd.to_numeric(results.iloc[6, 1:])
    euclidean_strict = pd.to_numeric(results.iloc[8, 1:])
    pearson = pd.to_numeric(results.iloc[9, 1:])
    pearson_wehbe = pd.to_numeric(results.iloc[10, 1:])
    pearson_strict = pd.to_numeric(results.iloc[12, 1:])

    # Set up subplots
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

    # Add empty outside plots to have one label for three subplots
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

def make_paper_plot(nonefile, evfile, savefile):
    nonedata = pd.read_csv(nonefile, sep='\t')
    print(nonedata)
    evdata = pd.read_csv(evfile, sep='\t')
    print(evdata)
    none1 = pd.to_numeric(nonedata.iloc[1, 1:])
    none2 = pd.to_numeric(nonedata.iloc[5, 1:])
    none3 = pd.to_numeric(nonedata.iloc[9, 1:])

    test1 = pd.to_numeric(nonedata.iloc[3, 1:])
    test2 = pd.to_numeric(nonedata.iloc[7, 1:])
    test3 = pd.to_numeric(nonedata.iloc[11, 1:])

    train1 = pd.to_numeric(evdata.iloc[1, 1:])
    train2 = pd.to_numeric(evdata.iloc[5, 1:])
    train3 = pd.to_numeric(evdata.iloc[9, 1:])

    print(none1)
    print(train1)
    print(test1)
    # Set up subplots
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(ncols=7, sharex='row', sharey="row")

    #plot1 = sns.violinplot(y=none1, color="darkorange", ax=axs[0])
    #plot1.set(xlabel='$R^2$', ylabel = "")
    #plot1.yaxis.setlim(-100000,400000)
    #plot1.xaxis.set_label_position('top')
    plot2 = sns.violinplot(y=train1, color="darkorange", ax=axs[0])
    plot2.set(xlabel='$R^2$', ylabel="")
    plot2.xaxis.set_label_position('top')
    plot3 = sns.violinplot(y=test1, color="darkorange", ax=axs[4])
    plot3.set(xlabel='$R^2$', ylabel="")
    plot3.xaxis.set_label_position('top')

    #add empty plot in the middle
    plotempty =sns.violinplot(y=[], color="darkorange", ax=axs[3])
    axs[3].grid(False)
    axs[3].axis('off')

    #plot4 = sns.violinplot(y=none2, color="darkorange", ax=axs[1])
    #plot4.set(xlabel='$EV$', ylabel="")
    #plot4.xaxis.set_label_position('top')
    plot5 = sns.violinplot(y=train2, color="darkcyan", ax=axs[1])
    plot5.set(xlabel='$EV$', ylabel="")
    plot5.xaxis.set_label_position('top')
    plot6 = sns.violinplot(y=test2, color="darkcyan", ax=axs[5])
    plot6.set(xlabel='$EV$', ylabel="")
    plot6.xaxis.set_label_position('top')
    #plot7 = sns.violinplot(y=none3, color="darkorange", ax=axs[2])
    #plot7.set(xlabel='$r^2 simple$', ylabel="")
    #plot7.xaxis.set_label_position('top')
    plot8 = sns.violinplot(y=train3, color="darkmagenta", ax=axs[2])
    plot8.set(xlabel='$r^2 simple$', ylabel="")
    plot8.xaxis.set_label_position('top')
    plot9 = sns.violinplot(y=test3, color="darkmagenta", ax=axs[6])
    plot9.set(xlabel='$r^2 simple$', ylabel="")
    plot9.xaxis.set_label_position('top')

    # Add empty outside plots to have one label for three subplots


    fig.add_subplot(162, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel("500 on train")
    fig.add_subplot(165, frameon=False)

    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel("500 on test")
    plt.ylabel("")
    plt.savefig(savefile)
    plt.show()
    

# get_results("WordsElmoEncoder")
# get_results("WordsRandomEncoder")

# get_results("PostsElmoEncoder")
# get_results("PostsRandomEncoder")
#
# get_results("HarryElmoEncoder", "Continuous")
# get_results("HarryRandomEncoder", "Continuous")

# get_results("AliceElmoEncoder", "Continuous")
# get_results("AliceRandomEncoder", "Continuous")
