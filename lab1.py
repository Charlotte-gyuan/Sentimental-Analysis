import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB


def main():
    f = open("training_dataset.txt")
    file = f.read()
    lines = file.split('\n')
    data =[]
    y =[]
    for line in lines:
        data.append(line[2:])
        y.append(line[0])


    f2 = open("test_dataset.txt")
    testfile = f2.read()
    testdata = testfile.split('\n')



    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(data).toarray()
    y = np.array(y)


    gnb = GaussianNB()
    scores = cross_val_score(gnb, x, y, cv=5, scoring='accuracy')
    print(scores)



    gnb.fit(x,y)
    x_test = vectorizer.transform(testdata).toarray()


    y_test = gnb.predict(x_test)
    for yi, doci in zip(y_test, testdata):
        print("{}\t{}".format(yi, doci))

    f.close()
    f2.close()


if __name__=='__main__':
    main()




