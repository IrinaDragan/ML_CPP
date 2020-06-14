#include "routines.h"
#include "time.h"
#include "windows.h"
#include "psapi.h"

list<point> dataframe, testframe;
int euclid_count = 0, manhattan_count = 0;

double myKNNFunction() {
    clock_t tStart = clock();
    dataframe.clear();
    testframe.clear();

    int euclidDistanceNeighbors = 10;
    int manhattanNeighbors = 10;
    std::ifstream in("KNNdata.txt"), test("KNNtest.txt"); // Read dataframe and testframe
    load_csv(in, dataframe);
    load_csv(test, testframe);
    list<point>::iterator i = dataframe.begin();   
   /* // Print frames
    cout << "DATA FRAME:" << endl << endl;
    for (; i != dataframe.end(); ++i)
        cout << *i << endl;

    cout << endl << "TEST FRAME:" << endl << endl;
    i = testframe.begin();
    for (; i != testframe.end(); ++i)
        cout << *i << endl;
    */
    // Compute distances and report predictions
    int temp;
    euclid_count = 0;
    manhattan_count = 0;
    i = testframe.begin();
    //cout << "ACTUAL \t EUCLIDEAN \t MANHATTAN" << endl;
    for (; i != testframe.end(); ++i)
    {
        //cout << "\t" << i->attributes.back() << "\t";
        temp = knn_classify(dataframe, *i, euclidDistanceNeighbors, sq_euclid_dist);
        //cout << temp << "\t";
        if (temp == i->attributes.back())
            ++euclid_count;
        temp = knn_classify(dataframe, *i, manhattanNeighbors, manhattan_dist);
        //cout << temp << endl;
        if (temp == i->attributes.back())
            ++manhattan_count;
    }
    in.close();
    test.close();

    return (double)(clock() - tStart) / CLOCKS_PER_SEC;
}

int main(int argc, char* argv[])
{   
    int nrOfIterations = 10;
    double mediumTime = 0;
  
    for (int i = 0; i < nrOfIterations; i++) {
        mediumTime+=myKNNFunction();
    }

    // Report accuracy
    cout << endl << "Euclid accuracy: ";
    cout << (1.0 * euclid_count / testframe.size()) * 100;
    cout << "%" << endl << "Manhattan accuracy: ";
    cout << (1.0 * manhattan_count / testframe.size()) * 100 << "%";
    cout << endl << endl;

    mediumTime /= nrOfIterations;
    cout << "Timp mediu de executie: " << mediumTime << "s"<<endl;

    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    SIZE_T virtualMemUsedByMe = pmc.PrivateUsage;
    cout << "Memorie utilizata: " << (double)virtualMemUsedByMe/1024/1024 <<"Mb"<<endl;
}