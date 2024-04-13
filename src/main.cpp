#include <iostream>
#include "mlpp.hpp"

using namespace DataAnalysis;
using namespace std;

int main()
{
    auto data = readCSV<double>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");

    // Print the matrix
    
    for (const auto& row : data) 
    {
        for (const auto& element : row) 
        {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    } 
    
    
    //header(data, 5);
    //bottom(data, 5);

    return 0;
}