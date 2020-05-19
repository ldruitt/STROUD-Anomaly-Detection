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

    strangetrain.sort()
    print(strangetrain)

    #produce a balanced test set by taking random signals (files) from ModeA, ModeB, ModeC, and ModeD folders (normal)
    #not used in training and adding random signals (files) from ModeM (anomalies)
    atestind = np.random.choice(alist.shape[0], 30, replace=False)      #get random indices of a
    atest = alist[atestind]                                             #get the full signals at the random indices
    atest = atest.reshape(30, 20000)                                    #retain shape

    btestind = np.random.choice(blist.shape[0], 30, replace=False)
    btest = blist[btestind]
    btest = btest.reshape(30, 20000)

    ctestind = np.random.choice(clist.shape[0], 30, replace=False)
    ctest = clist[ctestind]
    ctest = ctest.reshape(30, 20000)

    dtestind = np.random.choice(dlist.shape[0], 30, replace=False)
    dtest = dlist[dtestind]
    dtest = dtest.reshape(30, 20000)

    mtestind = np.random.choice(mlist.shape[0], 30, replace=False)
    mtest = mlist[mtestind]
    mtest = mtest.reshape(30, 20000)

    test = np.array([])                                         #test data of randomly distributed signals from all files
    test = np.append(test, atest)
    test = np.append(test, btest)
    test = np.append(test, ctest)
    test = np.append(test, dtest)
    test = np.append(test, mtest)
    test = test.reshape(150, 20000)                             #retain shape
    np.random.shuffle(test)                                     #retain randomness

    anomind = np.zeros(30, dtype=np.int)                        #get the indices of the anomalies
    for i, anom in enumerate(mtest):
        for j, elem in enumerate(test):
            if np.array_equiv(anom, elem):
                anomind[i] = j
                break
    print(anomind)

    #run each signal through FFT and keep the transformed points a test data
    label = np.zeros((150,20000))
    for i in range(150):
        fft = np.fft.fft(test[i])    # compute fft
        label[i] = fft

    #For each data point in the test set, compute the LOF with respect to the points in the training set
    #Form a list of LOF (strangeness values) for the test set. DO NOT SORT THIS ONE! Preserve the order you had in the
    # test set
    strangetest = np.zeros(150)
    k = 18
    neigh = NearestNeighbors(n_neighbors=k, p=2)
    neigh.fit(train)                                                        #compute with respect to training set
    for i, qpoint in enumerate(label):
        minpts = neigh.kneighbors([qpoint], n_neighbors=k)                  #reachability distance of q
        lrdq = (k)/np.sum(minpts[0][:][:])                                #lrd(q) = minpts/ sum(reachdist(q,p))
        lrdp = 0
        for j, index in enumerate(minpts[1][0][:]):
            if j == 0:
                continue
            reachdistp = neigh.kneighbors([label[index]], n_neighbors=k)    #reachability distances of p1
            lrdp1 = (k)/np.sum(reachdistp[0][:][:])                         #lrd(p1) = minpts/ sum(reachdist(p1,p))
            lrdp += lrdp1                                                   #sum(lrd(p))
        lof = 1/(k) *(lrdp/lrdq)                                            #lof(q) = 1/minpts * sum(lrd(p)/lrd(q))
        strangetest[i] = lof                                                #add to strangeness list

    print(strangetest)

    #For each measure on strangeness test find:
    #b = number of measures in train strangeness higher or equal to the test strangeness point
    #p-value of test point = b/N+1, N= size of strangeness list
    #list of p values
    plist = np.zeros(150)                            #list of p values
    for i, measure in enumerate(strangetest):
        strangetrain = np.append(strangetrain, measure)
        b = np.sum(strangetrain >= measure)         #b = number of signals in strangeness training that are greater than the test strangenesas point
        p = (b)/(strangetrain.size + 1)                                 #b/N+1, N = size of strange train
        plist[i] = p                                #add to list of p values
        strangetrain = np.delete(strangetrain, measure)

    print(plist)
    #With the list of test p-values, you can compute the AUC of the corresponding ROC
    #compute the AUC
    anoms = np.ones(150)
    np.put(anoms, anomind, 0)                           #labels of anomalies, 0 where anomaly exists

    auc = roc_auc_score(anoms, plist)                   #labels and probabilities fed to auc roc score
    print(auc)

    #For different training/test sets compute AUC
    #Compare AUC for different choices of k
    #Pick a winner k and use it for your next step




if __name__ == '__main__':
    main()