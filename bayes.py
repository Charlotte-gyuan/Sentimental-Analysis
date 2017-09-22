import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB


def main():
    with open("training_dataset.txt",'r') as f:
        file = f.read()
    lines = file.split('\n')
    data =[]
    y =[]
    for line in lines:
        data.append(line[2:])
        y.append(line[0])


    with open("test_dataset.txt") as f:
        testfile = f.read()
    testdata = testfile.split('\n')



    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(data).toarray()
    y = np.array(y)


    gnb = GaussianNB()

    scores = cross_val_score(gnb, x, y, cv=5, scoring='f1_weighted')#accuracy be default
    print(scores)
    print("Means: ",scores.mean())



    gnb.fit(x,y)
    x_test = vectorizer.transform(testdata).toarray()


    y_test = gnb.predict(x_test)

    print(y_test)
    # for yi, doci in zip(y_test, testdata):
    #     print("{}\t{}".format(yi, doci))



if __name__=='__main__':
    main()




