#include <iostream>

#include "mlpp.hpp"

using namespace std;
using namespace DataAnalysis;

// Mini tests for data analysis unit
int main() 
{
    auto data = read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
    auto cData = matrix_converter<float>(data);
    // Print data
    cout << "========================================" << endl;
    displayAll(data);
    cout << "========================================" << endl;
    vector<int> rows{0, 10, 20, 30, 40, 50, 60, 70, 80, 90};
    displayRows(data, rows);
    cout << "========================================" << endl;
    vector<int> cols{1, 3, 5, 7};
    displayColumns(data, cols);
    //vector<int> pos1 {100, 8};
    //auto el = find_by_pos(data, pos1);

    vector<string> testR{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    vector<string> testC(data.size(), "0");
    matrix_formatter<string>(data, COLUMN, 12, testC);
    // auto pos = find<string>(data, "http://www.hatfield-saunders.net/");
    // auto pos = find<string>(data, "2020-03-11");

    cout << "========================================" << endl;
    // cout << el << endl;
    // cout << pos[0] << ", " << pos[1] << endl;
    displayHead(data, 5);
    displayBottom(data, 5);

    return 0;
}
