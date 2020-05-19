import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

def main():
    #create Normal baseline from random samples of signal folders ABCD
    adirectory = 'C:/Users/Numpy BB/Desktop/cs484/HW4_Druitt_00956434/src/ModeA'    #directories of files(signals)
    bdirectory = 'C:/Users/Numpy BB/Desktop/cs484/HW4_Druitt_00956434/src/ModeB'
    cdirectory = 'C:/Users/Numpy BB/Desktop/cs484/HW4_Druitt_00956434/src/ModeC'
    ddirectory = 'C:/Users/Numpy BB/Desktop/cs484/HW4_Druitt_00956434/src/ModeD'
    mdirectory = 'C:/Users/Numpy BB/Desktop/cs484/HW4_Druitt_00956434/src/ModeM'
    twdirectory = 'C:/Users/Numpy BB/Desktop/cs484/HW4_Druitt_00956434/src/TestWT'

    #get list of signals, distribute randomly
    alist = np.array([])                                    #list of signals in A
    for filename in os.listdir(adirectory):
        if filename.endswith('.txt'):
            adat = np.loadtxt(adirectory + "/" + filename)
            alist = np.append(alist, adat)
            continue
        else:
            continue

    alist = alist.reshape(100, 20000)
    aind = np.random.choice(alist.shape[0], 30,  replace=False)
    abase = alist[aind]

    blist = np.array([])                                    #list of signals in B
    for filename in os.listdir(bdirectory):
        if filename.endswith('.txt'):
            bdat = np.loadtxt(bdirectory + "/" + filename)
            blist = np.append(blist, bdat)
            continue
        else:
            continue

    blist = blist.reshape(100, 20000)
    bind = np.random.choice(blist.shape[0], 30,  replace=False)
    bbase = blist[bind]

    clist = np.array([])                                    #list of signals in C
    for filename in os.listdir(cdirectory):
        if filename.endswith('.txt'):
            cdat = np.loadtxt(cdirectory + "/" + filename)
            clist = np.append(clist, cdat)
            continue
        else:
            continue

    clist = clist.reshape(100, 20000)
    cind = np.random.choice(clist.shape[0], 30,  replace=False)
    cbase = clist[cind]

    dlist = np.array([])                                    #list of signals in D
    for filename in os.listdir(ddirectory):
        if filename.endswith('.txt'):
            ddat = np.loadtxt(ddirectory + "/" + filename)
            dlist = np.append(dlist, ddat)
            continue
        else:
            continue

    dlist = dlist.reshape(99, 20000)
    dind = np.random.choice(dlist.shape[0], 30,  replace=False)
    dbase = dlist[dind]

    mlist = np.array([])                                    #list of signals in M
    for filename in os.listdir(mdirectory):
        if filename.endswith('.txt'):
            mdat = np.loadtxt(mdirectory + "/" + filename)
            mlist = np.append(mlist, mdat)
            continue
        else:
            continue

    mlist = mlist.reshape(99, 20000)

    twlist = np.array([])                                    #list of signals in M
    for filename in os.listdir(twdirectory):
        if filename.endswith('.txt'):
            twdat = np.loadtxt(twdirectory + "/" + filename)
            twlist = np.append(twlist, twdat)
            continue
        else:
            continue

    twlist = twlist.reshape(499, 20000)

    normalbase = np.array([])                                       #base of randomly distributed normal signals from all files
    normalbase = np.append(normalbase, abase)
    normalbase = np.append(normalbase, bbase)
    normalbase = np.append(normalbase, cbase)
    normalbase = np.append(normalbase, dbase)
    normalbase = np.reshape(normalbase, (120, 20000))               #retain shape
    np.random.shuffle(normalbase)                                   #retain randomness

    #Run each signal through single sided FFT to create the baseline/training data
    train = np.zeros((120,20000))
    for i in range(120):
        fft = np.fft.fft(normalbase[i])            #compute fft
        train[i] = fft

    #Compute the LOF of each point in the baseline with respect to the other points in the baseline
    # Put those LOF measures in a list and sort that list in ascending order to create strangeness training test
    strangetrain = np.zeros(120)
    k = 18
    neigh = NearestNeighbors(n_neighbors=k, p=2)
    neigh.fit(train)
    for i, qpoint in enumerate(train):
        minpts = neigh.kneighbors([qpoint], n_neighbors=k)  # reachability distances of q to nearest neighbors
        lrdq = (k) / np.sum(minpts[0][:][:])            # lrd(q) = minpts/ sum(reachdist(q,p))
        lrdp = 0
        for j, index in enumerate(minpts[1][0][:]):
            reachdistp = neigh.kneighbors([train[index]], n_neighbors=k)  # reachability distances of this p
            lrdp1 = (k) / np.sum(reachdistp[0][:][:])                 # lrd(p1) = minpts/ sum(reachdist(p,nearest of p))
            lrdp += lrdp1                                                 # sum(lrd(p))
        lof = 1 / (k) * (lrdp / lrdq)                   # lof(q) = 1/minpts * sum(lrd(p)/lrd(q))
        strangetrain[i] = lof                               # add to strangeness list
    np.sort(strangetrain)
    print(strangetrain)

    #run each signal through FFT and keep the transformed points a test data
    label = np.zeros((499,20000))
    for i in range(499):
        fft = np.fft.fft(twlist[i])    # compute fft
        label[i] = fft

    #For each data point in the test set, compute the LOF with respect to the points in the training set
    #Form a list of LOF (strangeness values) for the test set. DO NOT SORT THIS ONE! Preserve the order you had in the
    # test set
    strangetest = np.zeros(499)
    k = 18
    neigh = NearestNeighbors(n_neighbors=k, p=2)
    neigh.fit(train)                                                        #compute with respect to training set
    for i, qpoint in enumerate(label):
        minpts = neigh.kneighbors([qpoint], n_neighbors=k)                  #reachability distance of q
        lrdq = (k)/np.sum(minpts[0][:][:])                                #lrd(q) = minpts/ sum(reachdist(q,p))
        lrdp = 0
        for j, index in enumerate(minpts[1][0][:]):
            reachdistp = neigh.kneighbors([label[index]], n_neighbors=k)    #reachability distances of p1
            lrdp1 = (k-1)/np.sum(reachdistp[0][:][:])                         #lrd(p1) = minpts/ sum(reachdist(p1,p))
            lrdp += lrdp1                                                   #sum(lrd(p))
        lof = 1/(k-1) *(lrdp/lrdq)                                            #lof(q) = 1/minpts * sum(lrd(p)/lrd(q))
        strangetest[i] = lof                                                #add to strangeness list

    print(strangetest)

    #For each measure on strangeness test find:
    #b = number of measures in train strangeness higher or equal to the test strangeness point
    #p-value of test point = b/N+1, N= size of strangeness list
    #list of p values
    plist = np.zeros(499)                            #list of p values
    for i, measure in enumerate(strangetest):
        strangetrain = np.append(strangetrain, measure)
        b = np.sum(strangetrain >= measure)         #b = number of signals in strangeness training that are greater than the test strangenesas point
        p = (b)/(strangetrain.size + 1)                                 #b/N+1, N = size of strange train
        plist[i] = p                                #add to list of p values
        strangetrain = np.delete(strangetrain, measure)

    print(plist)
    f = open('output.txt','w')
    for prob in plist:
        f.write(str(prob) + '\n')

    f.close()




if __name__ == '__main__':
    main()