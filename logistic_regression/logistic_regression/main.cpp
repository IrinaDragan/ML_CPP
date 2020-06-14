using namespace std;
#include <iostream>
#include "logistic_regression.h"
#include "routines.h"
#include "time.h"
#include "windows.h"
#include "psapi.h"

double accuracy;

double myKNNFunction() {
    clock_t tStart = clock();
    
    std::vector<std::vector<double>> trainX;
    std::vector<unsigned long int> trainY;

    std::vector<std::vector<double>> testX;
    std::vector<unsigned long int> testY;

    std::ifstream in("train.txt"), test("test.txt"); // Read dataframe and testframe
    list<point> dataframe, testframe;
    load_csv(in, dataframe);
    load_csv(test, testframe);
    list<point>::iterator i = dataframe.begin();

    // Citim datele de train
    for (; i != dataframe.end(); ++i) {
        list<double> currentList=i->attributes;
        int currentListSize = currentList.size();
        list<double>::iterator it = currentList.begin();
        vector<double>tempVector;
        double tempY;
        for (int j = 0; j < currentListSize; j++) {
            if (j == currentListSize - 1) {
                tempY = *it;
            }
            else {
                tempVector.push_back(*it);
            }
            advance(it, 1);
        }
        trainX.push_back(tempVector);
        trainY.push_back(tempY);
    }
    //Citim datele de test
    i = testframe.begin();

    for (; i != testframe.end(); ++i) {
        list<double> currentList = i->attributes;
        int currentListSize = currentList.size();
        list<double>::iterator it = currentList.begin();
        vector<double>tempVector;
        double tempY;
        for (int j = 0; j < currentListSize; j++) {
            if (j == currentListSize - 1) {
                tempY = *it;
            }
            else {
                tempVector.push_back(*it);
            }
            advance(it, 1);
        }
        testX.push_back(tempVector);
        testY.push_back(tempY);
    }

    logistic_regression lg(trainX, trainY, NODEBUG);
    lg.fit();
    // Save the model
    lg.save_model("model.json");

    std::vector<unsigned long int>::iterator jy = testY.begin();

    int countGoodPredicted = 0;
    for (std::vector<vector<double>>::iterator j = testX.begin(); j != testX.end(); ++j) {
        std::map<unsigned long int, double> probabilities = lg.predict(*j);
        int predicted;
        double isZero = probabilities[0];
        double isOne = probabilities[1];
        if (isZero > isOne) {
            predicted = 0;
        }
        else {
            predicted = 1;
        }
        if (predicted == *jy) {
            countGoodPredicted++;
        }
        ++jy;
    }
    accuracy = (double) countGoodPredicted / (testX.size());

    return (double)(clock() - tStart) / CLOCKS_PER_SEC;
}

int main(int argc, char* argv[])
{
    int nrOfIterations = 10;
    double mediumTime = 0;

    for (int i = 0; i < nrOfIterations; i++) {
        mediumTime += myKNNFunction();
    }

    // Report accuracy
    cout << "Acuracy: " << accuracy * 100 << "%" << endl;

    mediumTime /= nrOfIterations;
    cout << "Timp mediu de executie: " << mediumTime << "s" << endl;

    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    SIZE_T virtualMemUsedByMe = pmc.PrivateUsage;
    cout << "Memorie utilizata: " << (double)virtualMemUsedByMe / 1024 / 1024 << "Mb" << endl;
}