#include <iostream>

#include "mlpp.hpp"

using namespace std;
using namespace MLPP;

// Mini tests for data analysis unit
int main() 
{ 
    auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
    auto data2 = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/organizations-100.csv");
    auto cData = DataAnalysis::matrix_converter<float>(data);
    // Print data
    cout << "========================================" << endl;
    //DataAnalysis::displayAll(data);
    cout << "========================================" << endl;
    vector<int> rows{0, 10, 20, 30, 40, 50, 60, 70, 80, 90};
    //DataAnalysis::displayRows(data, rows);
    cout << "========================================" << endl;
    vector<int> cols{2, 4, 6, 8};
    //DataAnalysis::displayColumns(data, cols);
    //vector<int> pos1 {100, 8};
    //auto el = find_by_pos(data, pos1);

    vector<string> testR{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    vector<string> testR1{"3", "4", "5", "3", "4", "5", "6", "7", "8", "9"};
    vector<string> testR2{"6", "7", "8", "3", "4", "5", "6", "7", "8", "9"};
    vector<string> testR3{"9", "10", "11", "3", "4", "5", "6", "7", "8", "9"};
    vector<string> testC(data.size(), "0");
    Mat2d<string> testAddR {{testR}, {testR1}, {testR2}, {testR3}};
    Mat2d<string> testAddC {{testR}, {testR1}, {testC}};
    Mat2d<int> testAddCandR {{rows}, {cols}};
    DataAnalysis::matrix_formatter<string>(data, ROWANDCOLUMN, testAddCandR ,testAddR);
    // auto pos = find<string>(data, "http://www.hatfield-saunders.net/");
    // auto pos = find<string>(data, "2020-03-11");

    cout << "========================================" << endl;
    DataAnalysis::displayRows(data, rows);
    //DataAnalysis::displayColumns(data, cols);
    // cout << el << endl;
    // cout << pos[0] << ", " << pos[1] << endl;
    //DataAnalysis::displayAll(data);
    DataAnalysis::displayHead(data, 20);
    //DataAnalysis::displayBottom(data, 5);

    return 0;
}


