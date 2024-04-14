#include <iostream>
#include "mlpp.hpp"

using namespace DataAnalysis;
using namespace std;

int main()
{
    auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
    auto pos1 = find<std::string>(data, "http://watkins.info/");
    auto pos2 = findAll<std::string>(data, "http://www.hatfield-saunders.net/");
        
    // Print the data
    /*
    for (auto &&row : data)
    {
        for (auto &&cell : row)
        {
            cout << cell << " ";
        }
        cout << endl;
    }
    */
    
    vector<int> testR {2, 5, 96, 98};
    //matrixFormatter(data, ROW, testR);

    vector<int> testC {1, 3, 5, 7, 12};
    //matrixFormatter(data, COLUMN, testC);

    Mat2d<int> testRaC {{testR}, {testC}};
    matrixFormatter(data, ROWANDCOLUMN, testRaC);

    cout << "========================================" << endl;
    header(data, 5);
    bottom(data, 5);

    return 0;
}