#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <type_traits>

//Creating 2d Matrix template, a "new" data type called Mat2d, which will create 2d Matrix
// based on parameter type
// Outside vector stores row data, inside vector store columns
template <typename T> using Mat2d = std::vector<std::vector<T>>;

namespace DataAnalysis
{
    // Declaring template method to read CSV files
    template <typename T> Mat2d<T> readCSV(const std::string& filePath)
    {
        // Create 2d matrix to store date in whatever type T comes
        Mat2d<T> data;

        // Open CSV file from path provided 
        std::ifstream file(filePath);

        // Check if file opened correctly
        if (!file.is_open())
        {
            // Display error and return empty data if file couldn't open
            std::cerr << "Error: Coun't open file provided at '" << filePath << "'" << std::endl;
            return data;
        }

        // Reach each line of the file, which by default begins as string
        std::string line;
        while (std::getline(file, line))
        {
            std::vector<T> row;
            std::stringstream ss(line);
            std::string cell;

            // Tokenize line by comma
            while (std::getline(ss, cell, ','))
            {
                // Checks cell string stream, and converts declared type T
                std::istringstream iss(cell);
                T val;
                iss >> val;
                row.push_back(val);
            }
            
            // Add converted row data into final 2d matrix data
            data.push_back(row);        
        }
        
        // Close file
        file.close();

        // Return final 2d matrix data of type T
        return data;     
    };

    // Find method, return first position found
    template <typename T> std::vector<int> find(const Mat2d<T>& dataMatrix, const T& desiredElemet)
    {
        // Creat position vector
        std::vector<int> pos {0,0};

        if (!dataMatrix.empty())
        {
            // Loop through matrix to check for desired element
            for (int row = 0; row < dataMatrix.size(); row++)
            {
                for (int column = 0; column < dataMatrix[row].size(); column++)
                {
                    if (dataMatrix[row][column] == desiredElemet)
                    {
                        pos[0] = row;
                        pos[1] = column;
                        return pos;
                    }              
                }           
            }  
        }
            

        // Until better error handling comes pos 0,0 will be the error 
        return pos;
    };

    // Find by pos method, return position on 2d Matrix

    // Find all method, return vector of positions
    template <typename T> Mat2d<int> findAll(const Mat2d<T>& dataMatrix, const T& desiredElemet)
    {
        // Creat position vector
        Mat2d<int> pos;

        if (!dataMatrix.empty())
        {    
            // Loop through matrix to check for desired element
            for (int row = 0; row < dataMatrix.size(); row++)
            {
                for (int column = 0; column < dataMatrix[row].size(); column++)
                {
                    if (dataMatrix[row][column] == desiredElemet)
                    {
                        std::vector<int> currentPos {row, column};
                        pos.push_back(currentPos);
                    }              
                }           
            }       
        }
        
        // Until better error handling comes empty position matrix will be the error
        return pos;
    };

    // Display first five rows method, prints first 5 rows + header row
    void header(const Mat2d<std::string>& dataMatrix, int rowsToPrint = 5)
    {
        if (!dataMatrix.empty())
        {
            for (int row = 0; row < rowsToPrint + 1; row++)
            {
                for (int columns = 0; columns < dataMatrix[row].size(); columns++)
                {
                    std::cout << dataMatrix[row][columns] << " ";
                }

                std::cout << "" << std::endl;
            }           
        }
        
        else  
        {
            std::cerr << "Data is empty" << std::endl;
        }
    }

    // Display last five colunms method, prints last five rows 
    void bottom(const Mat2d<std::string>& dataMatrix, int rowsToPrint = 5)
    {
        if (!dataMatrix.empty())
        {
            for (int row = dataMatrix.size() - rowsToPrint; row < dataMatrix.size(); row++)
            {
                for (int columns = 0; columns < dataMatrix[row].size(); columns++)
                {
                    std::cout << dataMatrix[row][columns] << " ";
                }

                std::cout << "" << std::endl;
            }           
        }
        
        else  
        {
            std::cerr << "Data is empty" << std::endl;
        }
    }

    // Remove categorical rows for better data analysis, return purely numerical 2d Matrix
}