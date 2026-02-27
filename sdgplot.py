"""
Music 422 
SDGPLOT.py - 
Plotting tool for BS.1116 Test Results

Required file structure
###############################
sdgplot.py
sample1/
    |- sample1_listener1.csv
    |- sample1_listener2.csv
sample2/
    |- sample2_listener1.csv
    |- sample2_listener2.csv
###############################

Call plot code with list of directory names and key identifier. For example for 128kbps
fig, ax = plotSDG(['sample1/', 'sample2/'], '128kbps')
plt.show()

-----------------------------------------------------------------------
 © 2009-26 Marina Bosi -- All rights reserved
-----------------------------------------------------------------------

"""
import matplotlib.pyplot as plt
import numpy as np
import csv, itertools, os, re

def parseCsv(fname):
    """
    PARSECSV - reads the SGD csv file and saves it as a dictionary. 
    Trusts that the format of the CSV is correct
    """
    with open(fname, "r") as files:
        reader = csv.reader(files)
        # Strip all header data
        reader = itertools.islice(reader, 5, None)
        keys = []
        values = []
        for data in reader:
            keys.append(data[0])
            values.append([float(data[1])])
    return dict(zip(keys, values))

def loadSdgCsvs(dir):
    """
    LOADSDGCSVS - loads all CSV files in a director. 
    Trusts that format of CSVs are correct
    args:
    dir - string, directory name. Example : 'dir/'
    """
    # Go through all files in directory
    # Create dictionary of all csv files
    dicts = []
    for file in os.listdir(dir):
        print(file)
        if file.endswith(".csv"):
            dicts.append(parseCsv(os.path.join(dir, file)))
    # Now combine all csv files into a list
    appDict = dicts[0]
    for k in appDict.keys():
        for i in range(1, len(dicts)):
            appDict[k].append(dicts[i][k][0])
    
    # Now we're going to rename the keys
    # And strip the filenames
    filename = re.search(r"(.*?)(?=_)", list(appDict.keys())[0]).group()
    kOld = []
    kNew = []
    for k in appDict.keys():
        match = re.search(r"(?<=\_)(.*?)(?=.wav)", k)
        if match is not None:
            kNew.append(match.group())
            kOld.append(k)

    for i in range(len(kOld)):
        appDict[kNew[i]] = appDict.pop(kOld[i])
    appDict['filename'] = filename
    return appDict

def plotSDG(dirs, key):
    """
    PLOTSDG - Plots SDG data from BS.1116 tests.
    See HW for correct filenames
    args:
    dirs - list of strings of directory names
    key - string, type of encoding to plot
    """
    sample = []
    # Load All CSVs and print for debugging
    for d in dirs:
        #for debugging
        print("\n\n"+ d+  "\n")
        sample.append(loadSdgCsvs(d))
     
    # # add overall results to sample results
    # totalKeys = sample[0].keys()
    # totals = dict().fromkeys(totalKeys)
    # totals['filename'] = "overall mean"
    # for k in list(totalKeys)[:-1]:
    #     vals = [item for s in sample for item in s[k]]
    #     totals[k] = vals
    # sample.append(totals)   
 
    nSamples = len(sample)
   
    # Plotting code
    fig, ax = plt.subplots()
    width = 0.35 # Looks good
    labels = [sample[i]['filename'] for i in range(nSamples)]
    x = np.arange(nSamples + 1)

    # Get barHeigh, barBottom, and barMeans
    barHeight = [np.abs(np.amax(sample[i][key]) - np.amin(sample[i][key])) for i in range(nSamples)]
    barBottom = [np.amin(sample[i][key]) for i in range(nSamples)]
    barMeans = [np.mean(sample[i][key]) for i in range(nSamples)]
    Height = np.abs( np.amin(barMeans) - np.max(barMeans))
    Bottom = np.amin(barMeans)
    Mean = np.mean(barMeans)
    
    #for debugging
    print()
    print("barMeans",barMeans)
    print("Height",Height)
    print("Bottom",Bottom)
    print("Mean",Mean)

    # # Add the overall
    labels.append('Overall')
    barHeight.append(Height )
    barBottom.append(Bottom)
    barMeans.append(Mean)


    # Define our rectangles
    rects = ax.bar(x+width, barHeight, width, barBottom, color='0.75', edgecolor='k', linewidth=1.5)
    ax.hlines(barMeans, x+width/2, x+3*width/2, colors='k', linewidth=2)
    ax.axhline(0.0, lw='1.5', c='k')
    ax.set(
        ylabel = 'SDG',
        title = '{}'.format(key),
        yticks = np.arange(-4, 1.5, 0.5),
        xticks = x + width,
        xticklabels = labels
    )
    ax.grid('True', axis='y', ls=':')
    return fig, ax

if __name__ == "__main__":
    fig, ax = plotSDG(['castanets/', 'glockenspiel/', 'harpsichord/', 'spgm/', 'MoonBears'], '128kbps')
    plt.savefig("128kbps.png")