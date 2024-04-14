#include <iostream>
#include "mlpp.hpp"

using namespace DataAnalysis;
using namespace std;

int main()
{
    auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
    auto cData = matrixConverter<float>(data);

    // Print the data
    /* 
    for (auto &row : cData)
    {
        for (auto &cell : row)
        {
            cout << " " << cell;
        }
        cout << endl;
    }

    for (size_t col = 0; col < data[100].size(); col++)
    {
        cout << col << " " << data[100][col] << endl;      
    } 
    */
    //vector<int> pos1 {100, 8};
    //auto el = findByPos(data, pos1);

    vector<string> testR {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    vector<string> testC (data.size(), "0");
    //matrixFormatter<string>(data, COLUMN, 98, testR);
    auto pos = find<string>(data, "http://www.hatfield-saunders.net/");
    //auto pos = find<string>(data, "2020-03-11");    

    cout << "========================================" << endl;
    //cout << el << endl;
    cout << pos[0] << ", " << pos[1] << endl;
    header(data, 5);
    bottom(data, 5);

    return 0;
}