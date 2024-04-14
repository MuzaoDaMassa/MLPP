#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
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

    // Method to parse CSV line with quoted fields and blank spaces
    std::vector<std::string> parseCSVLine(const std::string &line)
    {
        std::vector<std::string> row;

        if (!line.empty())
        {
            std::stringstream ss(line);
            std::string cell;

            // Tokenize line by comma
            while (std::getline(ss, cell, ','))
            {
                // Check if cell starts with a quote
                if (cell.front() == '"')
                {
                    std::string quotedCell;
                    quotedCell += cell;

                    // Keep reading until we find the closing quote
                    while (ss && cell.back() != '"')
                    {
                        std::getline(ss, cell, ',');
                        quotedCell += "," + cell;
                    }

                    // Fill out row vector with parsed cells with quotes removed
                    quotedCell = quotedCell.substr(1, quotedCell.size() - 2);
                    row.push_back(quotedCell);
                }
                else if (cell.find(' ') == std::string::npos)
                {
                    // Fill out row vector with parsed cells
                    // Using istringstream for cells with no space works better
                    std::istringstream iss(cell);
                    std::string val;
                    iss >> val;
                    row.push_back(val);
                }   
                else
                {
                    row.push_back(cell);
                }           
            }

            // Return filled out row
            return row;
        }
        // Display error and return empty row
        std::cerr << "Error: Line provided is empty" << std::endl;
        return row;
    };

    // Method to read CSV files into 2d string matrix
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

        // Reach each line of the file
        std::string line;
        while (std::getline(file, line))
        {
            // Parse CSV line and add it to the data matrix
            std::vector<std::string> row = parseCSVLine(line);
            data.push_back(row);
        }
        
        // Close file
        file.close();

        // Return data in matrix
        return data;     
    };

    // Method to convert string to differnt data type, returns new 2d matrix with passed type
    template <typename T> Mat2d<T> matrixConverter(const Mat2d<std::string>& stringMatrix)
    {   
        // Create 2d matrix of new typing
        Mat2d<T> convertedMatrix;

        if (!stringMatrix.empty())
        {
            // Loop through all elements changing their typing
            for (const auto &row : stringMatrix)
            {
                std::vector<T> convertedRow;

                for (const auto &element : row)
                {
                    // Convert string to type T
                    std::stringstream ss(element);
                    T convertedElement;
                    ss >> convertedElement;
                    convertedRow.push_back(convertedElement);
                }

                // Replace string row with newly converted row to new 2d matrix
                convertedMatrix.push_back(convertedRow);
            }

            // Return new data matrix of type T
            return convertedMatrix;
        }
        else
        {
            // Display error to inform data is empty
            std::cerr << "Error: Data matrix is empty!" << std::endl;
            return convertedMatrix;
        }
        
    };

    // Method to format data matrix by removing rows and or columns for better data analysis
	// Template for removing single row or column
    template <typename T> void matrixFormatter(Mat2d<T>& dataMatrix, FORMATTER f, const int& toBeRemoved)
    {	
        if (!dataMatrix.empty())
        {
			if (f == COLUMN)
			{
				if (toBeRemoved >= 0)
                {
                    // Iterate over each row and erase the element at colIndex
                    for (auto &row : dataMatrix)
                    {
                        if (toBeRemoved < row.size())
                        {
                            row.erase(row.begin() + toBeRemoved);
                            return;
                        }
                    }    
                    
                    // Display error to inform invalid index to remove
                    std::cerr << "Error: Invalid index of column to remove" << std::endl;        
                }    
                // Display error to inform invalid index to remove
                std::cerr << "Error: Invalid index of column to remove" << std::endl;         
			}
			else if(f == ROW)
			{
				if (toBeRemoved >= 0 && toBeRemoved <= dataMatrix.size())
                {
                    // Delete selected row from matrix 
                    dataMatrix.erase(dataMatrix.begin() + toBeRemoved);
                    return;
                }
                // Display error to inform invalid index to remove
                std::cerr << "Error: Invalid index of row to remove" << std::endl;
			}        
            else
            {
                // Display error to inform wrong formatter enum was passed
                std::cerr << "Error: Wrong formatter passed (2nd Paramater)" << std::endl;
            }                         
        }    
        // Display error to inform data is empty
        std::cerr << "Error: Data Matrix is empty" << std::endl;
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
                            if (colIndex < dataMatrix[row].size())
							{
								dataMatrix[row].erase(dataMatrix[row].begin() + colIndex);
							}	
                        }
                    }
                    return;
                }
                // Display error to inform position data is empty
                std::cerr << "Error: No rows or columns to remove were found" << std::endl;
			}
            else if (f == ROW)
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
                    return;
				}
                // Display error to inform position data is empty
                std::cerr << "Error: No rows or columns to remove were found" << std::endl;
            }            
			else
			{				
                // Display error to inform wrong formatter enum was passed
                std::cerr << "Error: Wrong formatter passed (2nd Paramater)" << std::endl;
			}		
		}		
        // Display error to inform data is empty
        std::cerr << "Error: Data Matrix is empty" << std::endl;
	};

	// Template for removing single or multilple rows and columns
	// racToRemove = rows and columns to remove, outside vector will hold rows to remove while inside will hold columns
	template <typename T> void matrixFormatter(Mat2d<T>& dataMatrix, FORMATTER f, const Mat2d<int>& racToRemove)
	{
        if (!dataMatrix.empty())
        {
            if (f == ROWANDCOLUMN)
            {
                if (!racToRemove.empty())
                {
                    matrixFormatter(dataMatrix, ROW, racToRemove[0]);
                    matrixFormatter(dataMatrix, COLUMN, racToRemove[1]);
                    return;
                }
                // Display error if no position data  was found
                std::cerr << "Error: No rows and columns to remove" << std::endl;
            }
            // Display error to inform wrong formatter enum was passed
            std::cerr << "Error: Wrong formatter passed (2nd Paramater)" << std::endl;
        }
    };

	// Template for adding single row or column
    template <typename T> void matrixFormatter(Mat2d<T>& dataMatrix, FORMATTER f, const int& indexToAdd, const std::vector<T>& dataToAdd)
    {
        if (!dataMatrix.empty())
        {
            if (f == COLUMN)
            {
                // Check if indexToAdd is valid and within bounds
                if (indexToAdd >= 0 && indexToAdd <= dataMatrix[0].size())
                {
                    for (int i = 0; i < dataToAdd.size(); i++)
                    {
                        // Insert dataToAdd[i] at selected index in each row
                        dataMatrix[i].insert(dataMatrix[i].begin() + indexToAdd, dataToAdd[i]);
                    }
                    return;
                }

                std::cerr << "Error: Invalid index to add for inserting column." << std::endl;
            }
            else if (f == ROW)
            {
                // Check if indexToAdd is valid and within bounds
                if (indexToAdd >= 0 && indexToAdd <= dataMatrix.size())
                {
                    dataMatrix.insert(dataMatrix.begin() + indexToAdd, dataToAdd);
                    return;
                }

                std::cerr << "Error: Invalid index to add for inserting row." << std::endl;
            }   
            else
            {
                // Display error to inform formatter enum error no data was found
                std::cerr << "Error: Wrong formatter passed (2nd Paramater)" << std::endl;
            }
        }
        // Display error if no data was found
        std::cerr << "Error: Data Matrix is empty" << std::endl;
    };

	// Template for adding multiple rows or columns
	// Template for adding multiple rows and columns

    // Method to search data matrix for first appearance of desired element, return first position found
    template <typename T> std::vector<int> find(const Mat2d<T>& dataMatrix, const T& desiredElement)
    {
        // Creat position vector
        std::vector<int> pos {0,0};

        if (!dataMatrix.empty())
        {
            // Loop through matrix to check for desired element         
            for (size_t row = 0; row < dataMatrix.size(); row++)
            {
                //std::cout << row << std::endl; 
                for (size_t col = 0; col < dataMatrix[row].size(); col++)
                {     
                    //std::cout << col << std::endl;     
                    if (dataMatrix[row][col] == desiredElement)
                    {
                        pos[0] = row;
                        pos[1] = col;
                        return pos;
                    }                             
                }   
            }

            // Return empty vector
            std::cerr << "Error: Desired element was not found" << std::endl;
            return pos;
        }

        // Return 0,0 vector
        std::cerr << "Error: Data Matrix is empty" << std::endl;
        return pos;
    };

    // Method to search data matrix by position, return element at requqested position on 2d Matrix
    template <typename T> T findByPos(const Mat2d<T>& dataMatrix, std::vector<int>& pos)
    {
        T element;

        if (!dataMatrix.empty())
        {
            element = dataMatrix[pos[0]][pos[1]];
            return element;
        }

        // Return empty element
        std::cerr << "Error: Data Matrix is empty" << std::endl;
        return element;       
    };

    // Method to search data matrix for all appearances of desired element, return vector of positions
    // Outside vecotor holds all rows indexes element was found, and inside vector holds all columns indexes element was found
    template <typename T> Mat2d<int> findAll(const Mat2d<T>& dataMatrix, const T& desiredElemet)
    {
        // Creat position vector
        Mat2d<int> pos;

        if (!dataMatrix.empty())
        {    
            // Loop through matrix to check for desired element
            for (int row = 0; row < dataMatrix.size(); row++)
            {
                for (int col = 0; col < dataMatrix[row].size(); col++)
                {
                    if (dataMatrix[row][col] == desiredElemet)
                    {
                        std::vector<int> currentPos {row, col};
                        pos.push_back(currentPos);
                    }              
                }           
            }    

            if (pos.empty())
            {
                // Return empty position matrix
                std::cerr << "Error: Element was not found" << std::endl;
                return pos; 
            }

            // Return found positions
            return pos;             
        }
        
        // Return empty position matrix
        std::cerr << "Error: Data Matrix is empty" << std::endl;
        return pos;
    };

    // Method to display first five rows method, Display first 5 rows + header row
    void header(const Mat2d<std::string>& dataMatrix, int rowsToDisplay = 5)
    {
        if (!dataMatrix.empty())
        {
            for (int row = 0; row < rowsToDisplay + 1; row++)
            {
                for (int col = 0; col < dataMatrix[row].size(); col++)
                {
                    std::cout << " " << dataMatrix[row][col];
                }

                std::cout << std::endl;
            }           
        }          
        else  
        {
            std::cerr << "Data is empty" << std::endl;
        }
    };

    // Mehtod to display last five colunms method, Display last five rows 
    void bottom(const Mat2d<std::string>& dataMatrix, int rowsToDisplay = 5)
    {
        if (!dataMatrix.empty())
        {
            for (int row = dataMatrix.size() - rowsToDisplay; row < dataMatrix.size(); row++)
            {
                for (int col = 0; col < dataMatrix[row].size(); col++)
                {
                    std::cout << " " << dataMatrix[row][col];
                }

                std::cout << std::endl;
            }           
        }           
        else  
        {
            std::cerr << "Data is empty" << std::endl;
        }
    };
    
    // Further methods to be implemented 
}