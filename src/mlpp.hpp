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
    enum FORMATTER 
    {
        ROW, 
        COLUMN,
        ROWANDCOLUMN
    };

    // Declaring method to read CSV files into 2d string matrix
    Mat2d<std::string> readCSV(const std::string& filePath)
    {
        // Create 2d matrix to store data
        Mat2d<std::string> data;

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
            std::vector<std::string> row;
            std::stringstream ss(line);
            std::string cell;

            // Tokenize line by comma
            while (std::getline(ss, cell, ','))
            {
                // Checks cell string stream, and pass through val to row
                std::istringstream iss(cell);
                std::string val;
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

    // Convert string to differnt data type, returns new 2d matrix with passed type
    template <typename T> Mat2d<T> matrixConverter(const Mat2d<std::string>& stringMatrix)
    {   
        // Create 2d matrix of new typing
        Mat2d<T> convertedMatrix;

        if (stringMatrix.empty())
        {
            std::cerr << "Passed matrix is empty!" << std::endl;
            return convertedMatrix;
        }

        else
        {
            // Loop through all elements changing their typing
            for (const auto& row : stringMatrix) 
            {
                std::vector<T> convertedRow;

                for (const auto& element : row) 
                {
                    // Convert string to type T
                    std::stringstream ss(element);
                    T convertedElement;
                    ss >> convertedElement;
                    convertedRow.push_back(convertedElement);
                }
                
                // Add converted element to new 2d matrix
                convertedMatrix.push_back(convertedRow);
            } 

            return convertedMatrix;
        }
        
    };

    // Format matrix by removing rows and or columns for better data analysis
	// Template for removing single row or column
    template <typename T> void matrixFormatter(Mat2d<T>& dataMatrix, FORMATTER f, const int& toBeRemoved)
    {	
        if (!dataMatrix.empty())
        {
			if (f == COLUMN)
			{
				// Iterate over each row and erase the element at colIndex
				for (auto &row : dataMatrix)
				{
					if (toBeRemoved <= row.size())
					{
						row.erase(row.begin() + toBeRemoved);
					}
				}
			}
			else
			{
				// Delete selected row from matrix 
                dataMatrix.erase(dataMatrix.begin() + toBeRemoved);
			}                                 
        }    
    };

	// Template for removing multiple rows or columns
	// rocToRemove = rows or columns to remove, vector will hold all rows or columns to remove
	template <typename T> void matrixFormatter(Mat2d<T>& dataMatrix, FORMATTER f, const std::vector<int>& rocToRemove)
	{
		if (!dataMatrix.empty())
		{
			if (f == COLUMN)
			{
				if (!rocToRemove.empty())
                {
                    // Iterate over each row and erase the element at colIndexexes
                    for (int row = 0; row < dataMatrix.size(); row++)
                    {
                        for (auto col = rocToRemove.rbegin(); col != rocToRemove.rend(); ++col)
                        {
                            size_t colIndex = *col;
                            if (colIndex <= dataMatrix[row].size())
							{
								dataMatrix[row].erase(dataMatrix[row].begin() + colIndex);
							}
							
                        }
                    }
                }
			}

			else
			{
				if (!rocToRemove.empty())
				{
					// Iterate through declared rows to be removed from matrix
					for (auto item= rocToRemove.rbegin(); item!= rocToRemove.rend(); ++item)
					{
						size_t rowIndex = *item;
						if (rowIndex < dataMatrix.size())
						{
							dataMatrix.erase(dataMatrix.begin() + rowIndex);
						}
					}
				}
			}		
		}		
	};

	// Template for removing single or multilple rows and columns
	// racToRemove = rows and columns to remove, outside vector will hold rows to remove while inside will hold columns
	template <typename T> void matrixFormatter(Mat2d<T>& dataMatrix, FORMATTER f, const Mat2d<int>& racToRemove)
	{
		if (f == ROWANDCOLUMN)
		{
			if (!racToRemove.empty())
			{
				matrixFormatter(dataMatrix, ROW, racToRemove[0]);
				matrixFormatter(dataMatrix, COLUMN, racToRemove[1]);
			}	
		}		
	};

	// Template for adding single row or column
	// Template for adding multiple rows or columns
	// Template for adding multiple rows and columns

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

    // Find by pos method, return element at requqested position on 2d Matrix
    template <typename T> T findByPos(const Mat2d<T>& dataMatrix, std::vector<int>& pos)
    {
        T element;

        if (!dataMatrix.empty())
        {
            element = dataMatrix[pos[0]][pos[1]];
            return element;
        }

        return element;       
    }

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
    
    // Further methods to be implemented 
}