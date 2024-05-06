/*
MIT License

Copyright (c) 2024 Murilo Salviato Pileggi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#pragma region Includes
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <random>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <valarray>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#pragma endregion

namespace MLPP 
{
    // Declaring 2d Matrix template, a "new" data type called Mat2d, which will
    // create 2d Matrix based on parameter type
    // Outside vector stores row data, inside vector store columns
    template <typename T> using Mat2d = std::vector<std::vector<T>>;
    // Declare 3d matrix template where each element is a vector
    template <typename T> using Mat3d = std::vector<Mat2d<T>>;  

    // Formatter utility enum to help with method overload
    enum Formatter { ROW, COLUMN, ROWANDCOLUMN };

    // Class that contains all methods that are needed for numeric computation
    class NumPP
    {
    public:
        // Method to transpose 2d matrix, [100, 50] -> [50, 100]
        template <typename T> 
        static Mat2d<T> transpose(const Mat2d<T>& to_transpose)
        {
            if (!to_transpose.empty()) {
                // Get original matrix dimensions
                size_t rows = to_transpose.size();
                size_t cols = rows > 0 ? to_transpose[0].size() : 0;

                // Create new 2d matrix object with reshaped dimensions
                Mat2d<T> t(cols, std::vector<T>(rows));
                // Transpose the original matrix to reshape it
                for (size_t i = 0; i < cols; i++) {              
                    for (size_t j = 0; j < rows; j++) {
                        t[i][j] = to_transpose[j][i];
                    }      
                }
                return t;
            }
            // Display error that informs data matrix is empty
            std::cerr << "Error: Matrix is empty" << std::endl;
            return Mat2d<T>(); // Return empty matrix
        }
        
        // Method that receives two 2d matrices and returns their dot product
        template <typename T> 
        static Mat2d<T> dot(const Mat2d<T>& a, const Mat2d<T>& b)
        {       
            if (!a.empty() || !b.empty()) {
                if (a[0].size() == b.size()) { 
                    // Create new 2d matrix to store result
                    Mat2d<T> result(a.size(), std::vector<T>(b[0].size())); 
                    for (size_t a_row = 0; a_row < a.size(); a_row++) {
                        //std::vector<T> nRow(result[a_row].size());
                        for (size_t b_col = 0; b_col < b[0].size(); b_col++) {
                            T r = 0; // Create smart pointer to store temporary calculation results
                            for (size_t k = 0; k < b.size(); k++) {
                                r += a[a_row][k] * b[k][b_col];
                            }      
                            result[a_row][b_col] = r;
                        }
                    }
                    return result; // Return matrix with new values
                }
                // Display error that informs matrix a column size is different to matrix b row size
                std::cerr << "Error: Incompatible shapes" << std::endl;
                return Mat2d<T>(); // Return empty matrix
            }
            // Display error that informs data matrix is empty
            std::cerr << "Error: Input empty" << std::endl;
            return Mat2d<T>(); // Return empty matrix
        }
        
        // Method that adds vector to each row of matrix 
        template <typename T>
        static Mat2d<T> add(const Mat2d<T>& mat, const std::vector<T>& vec)
        {
            if (mat.empty() || vec.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<T>(); // Return empty matrix
            }

            Mat2d<T> result(mat.size(), std::vector<T>(vec.size()));

            for (size_t i = 0; i < result.size(); i++) {
                for (size_t j = 0; j < result[i].size(); j++) {
                    result[i][j] = mat[i][j] + vec[j];
                }
            }

            return result;
        }

        // Method to multiply corresponding elements of 2 2d matrices
        // Matrices must be same size
        template <typename T> 
        static Mat2d<T> mat_mul_matching_elements(const Mat2d<T>& a, const Mat2d<T>& b)
        {
            if (a.empty() || b.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<T>(); // Return empty matrix
            }

            if (a.size() != b.size() || a[0].size() != b[0].size()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Incompatible shapes" << std::endl;
                return Mat2d<T>(); // Return empty matrix
            }

            Mat2d<T> result(a.size(), std::vector<T>(a[0].size()));

            for (size_t row = 0; row < a.size(); row++) {
                for (size_t col = 0; col < a[row].size(); col++) {
                    result[row][col] = a[row][col] * b[row][col];
                }
            }

            return result;
        }
         
        // Returns sum of all multiplications, matrices must have same size
        template <typename T> 
        static T sum_mat_mul_matching_elements(const Mat2d<T>* a, const Mat2d<T>* b)
        {
            if (a->empty() || b->empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return 0; // Return 0
            }

            T r = 0; // Create variable to store results
            // Iterate through every element in a, and multiply by matching elemnt in b
            for (size_t i = 0; i < a->size(); i++) {
                for (size_t j = 0; j < (*a)[i].size(); j++) {
                    r += (*a)[i][j] * (*b)[i][j];
                }
            }

            return r;
            
        } 

        // Method that multiplies given number with all elements in matrix
        template <typename T>
        static Mat2d<T> scalar_mat_mul(const Mat2d<T>& mat, const T& scalar)
        {
            if (mat.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<T>(); // Return empty matrix
            }

            Mat2d<T> result(mat.size(), std::vector<T>(mat[0].size()));

            for (size_t i = 0; i < result.size(); i++) {
                for (size_t j = 0; j < result[i].size(); j++) {
                    result[i][j] = mat[i][j] * scalar;
                }
            }

            return result;
        }
        
        // Overload for different input and output type
        template <typename T, typename R>
        static Mat2d<R> scalar_mat_mul(const Mat2d<T>& mat, const R& scalar)
        {
            if (mat.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<R>(); // Return empty matrix
            }

            Mat2d<R> result(mat.size(), std::vector<R>(mat[0].size()));

            for (size_t row = 0; row < result.size(); row++) {
                for (size_t col = 0; col < result[row].size(); col++) {
                    result[row][col] = static_cast<R>(mat[row][col]) * scalar;
                }
            }

            return result;
        }

        // Method that subtracts every element in mat by given number
        template <typename T>
        static Mat2d<T> scalar_mat_sub(const Mat2d<T>& mat, const T& scalar)
        {
            if (mat.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<T>(); // Return empty matrix
            }

            Mat2d<T> result(mat.size(), std::vector<T>(mat[0].size()));

            for (auto &row : result) {
                for (auto &el : row) {
                    el = (el - scalar);
                }
            }

            return result;
        }
        
        // Overload that subtracts given number by every element in matrix
        template <typename T>
        static Mat2d<T> scalar_mat_sub(const T& scalar, const Mat2d<T>& mat)
        {
            if (mat.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<T>(); // Return empty matrix
            }

            Mat2d<T> result(mat.size(), std::vector<T>(mat[0].size()));

            for (auto &row : result) {
                for (auto &el : row) {
                    el = (scalar - el);
                }
            }

            return result;
        }

        // Method that multiplies given number with all elements in vector
        template <typename T>
        static std::vector<T> scalar_vec_mul(const std::vector<T>& vec, const T& scalar)
        {
            if (vec.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return std::vector<T>(); // Return empty vector
            }

            std::vector<T> result(vec.size());
            for (size_t i = 0; i < result.size(); i++) {
                result[i] = vec[i] * scalar;
            }
            
            return result;        
        }

        // Method to subtract two matrices
        template <typename T>
        static Mat2d<T> subtract(const Mat2d<T>& a, const Mat2d<T>& b)
        {
            if (a.empty() || b.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<T>(); // Return null pointer        
            }

            if ((a.size() != b.size()) || (a[0].size() != b[0].size())) {
                // Display error that informs matrix a column size is different to matrix b row size
                std::cerr << "Error: Incompatible shapes" << std::endl;
                return Mat2d<T>(); // Return null pointer
            }

            Mat2d<T> result(a.size(), std::vector<T>(a[0].size()));

            for (size_t row = 0; row < result.size(); row++) {
                for (size_t col = 0; col < result[row].size(); col++) {
                    result[row][col] = a[row][col] - b[row][col];
                }
            }

            return result;
        }
        
        // Overload to subtract two vectors
        template <typename T>
        static std::vector<T> subtract(const std::vector<T>& a, const std::vector<T>& b)
        {
            if (a.empty() || b.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return std::vector<T>(); // Return null pointer        
            }

            if (a.size() != b.size()) {
                // Display error that informs vectors are of different sizes
                std::cerr << "Error: Incompatible shapes" << std::endl;
                return std::vector<T>(); // Return null pointer
            }

            std::vector<T> result(a.size());

            for (size_t i = 0; i < result.size(); i++)
            {
                result[i] = a[i] - b[i];
            }

            return result;         
        }

        // Method that apply tanh function to every element in 2d matrix
        template <typename T>
        static Mat2d<double> tanh(const Mat2d<T>& mat)
        {
            Mat2d<double> tanhMat(mat.size(), std::vector<double>(mat[0].size()));

            for (size_t row = 0; row < tanhMat.size(); row++) {
                for (size_t col = 0; col < tanhMat[row].size(); col++) {
                    tanhMat[row][col] = std::tanh(mat[row][col]);
                }
            }
            return tanhMat;
        }  

        // Method that generates random 2d matrix 
        template <typename T>
        static Mat2d<T> rand(const size_t& rows, const size_t& cols, const double& mean = 0.0, const double& stddev = 1.0)
        {
            // Create new 2d matrix object with specified parameters to later return
            Mat2d<T> result(rows, std::vector<T>(cols));

            // Initialize random generator - Normal Distribution, receives mean and standard deviation as paramaters
            std::random_device rd;
            std::mt19937 generator(rd());
            std::normal_distribution<double> distribution(mean, stddev);

            // Fill matrix with random values
            for (size_t rows = 0; rows < result.size(); rows++) {
                for (size_t cols = 0; cols < result[rows].size(); cols++) {
                    result[rows][cols] = static_cast<T>(distribution(generator));
                }
            }

            // Return random 2d matrix
            return result;
        }

        // Method that generates vector of size N filled with 0
        template <typename T>
        static std::vector<T> zeros(const int& size)
        {
            return std::vector<T>(size, 0);
        }

        // OVerload of zeros that generates 2d matrix of size R x C filled with 0
        template <typename T>
        static Mat2d<T> zeros(const size_t& rows, const size_t& cols)
        {
            return Mat2d<T>(rows, std::vector<T>(cols, 0));
        }

        // Method that generates vector of size N filled with 1
        template <typename T>
        static std::vector<T> ones(const int& size)
        {
            return std::vector<T>(size, 1);
        }

        // OVerload of zeros that generates 2d matrix of size R x C filled with 1
        template <typename T>
        static Mat2d<T> ones(const size_t& rows, const size_t& cols)
        {
            return Mat2d<T>(rows, std::vector<T>(cols, 1));
        }

        // Method that generates square 2d matrix
        template <typename T>
        static Mat2d<T>* gen_square_matrix(const size_t& size, const T& value = 0)
        {
            // Create new 2d matrix object with specified parameters to later return
            Mat2d<T>* result = new Mat2d<T>(size, std::vector<T>(size));

            // Fill matrix with specified value
            for (size_t rows = 0; rows < result->size(); rows++) {
                for (size_t cols = 0; cols < (*result)[rows].size(); cols++) {
                    (*result)[rows][cols] = value;
                }
            }

            // Return random 2d matrix
            return result;
        }

        // Method that returns sum of all elements in vector
        template <typename T>
        static T get_sum_of_vector(const std::vector<T>& vec)
        {
            T r;
            for (size_t i = 0; i < vec.size(); i++) {
                r += vec[i];
            }
            return r;
        }

        // Get sum of vector overload for diffferent type input and output
        template <typename T, typename R>
        static R get_sum_of_vector(const std::vector<T>& vec)
        {
            R r;
            for (size_t i = 0; i < vec.size(); i++) {
                r += static_cast<R>(vec[i]);
            }
            return r;
        }

        // Method that collapses 2d matrix into vector with each element
        // being the sum of each row or each column from 2d matrix, 0 = rows, 1 = columns
        template <typename T> 
        static std::vector<T> sum(const Mat2d<T>& mat, const size_t& axis = 0)
        {
            if (mat.empty()) {
                std::cerr << "Error: Matrix is empty" << std::endl;
                return std::vector<T>();
            }

            if (axis == 1) {
                // Get sum of each column into vector
                std::vector<T> result(mat[0].size(), 0); // Create vector to store result
                for (size_t i = 0; i < result.size(); i++) {
                    for (size_t j = 0; j < mat.size(); j++) {
                        result[i] += mat[j][i];
                    }
                }

                return result;
            }

            std::vector<T> result(mat.size()); // Create vector to store result

            for (size_t i = 0; i < result.size(); i++) {
                result[i] = get_sum_of_vector<T>(mat[i]);
            }

            return result;
        }

        // Method that elevates each element of given matrix to the power of n
        template <typename T>
        static std::vector<double> power(const std::vector<T>* vecPtr, const size_t& power = 2)
        {
            if (vecPtr == nullptr) {
                std::cerr << "Error: Matrix pointer is null" << std::endl;
                return std::vector<double>(); // Return empty matrix
            }

            std::vector<double> results(vecPtr->size());

            for (size_t i = 0; i < results.size(); i++)
            {
                results[i] = std::pow((*vecPtr)[i], power);
            }
            
            return results;
        }

        // Overload for 2d matrix
        template <typename T>
        static Mat2d<double> power(const Mat2d<T>& mat, const size_t& power = 2)
        {
            if (mat.empty()) {
                std::cerr << "Error: Matrix pointer is null" << std::endl;
                return Mat2d<double>(); // Return empty matrix
            }
            
            Mat2d<double> results(mat.size(), std::vector<double>(mat[0].size()));

            for (size_t row = 0; row < results.size(); row++) {
                for (size_t col = 0; col < results[row].size(); col++) {
                    results[row][col] = std::pow(mat[row][col], power);
                }
            }

            return results;
        }
       
        // Method that returns center coordinates of given 2d matrix
        template <typename T>
        static std::vector<int> get_center(const Mat2d<T>* matPtr)
        {
            if (matPtr == nullptr) {
                std::vector<int> pos {0,0};
                // Display error that informs data matrix is empty
                std::cerr << "Error: Null matrix pointer" << std::endl;
                return pos; // Return empty vector
            }

            std::vector<int> pos {matPtr->size()/2, (*matPtr)[matPtr->size()/2].size()/2}; // Create vector pointer to store position     
            return pos;
        }

        // Overload to get center of 1d matrix, i.e, a vector
        template <typename T>
        static int get_center(const std::vector<T>* vecPtr)
        {
            return vecPtr->size()/2;
        }

        // Method that finds and store maximun value in 2d matrix
        template <typename T>
        static T find_max_value(const Mat2d<T>& mat)
        {
            if (mat.empty()) {
                std::cerr << "Error: Input empty" << std::endl;
                return 0;
            }

            T result = 0;

            for (size_t i = 0; i < mat.size(); i++) {
                T max = *std::max_element(mat[i].begin(), mat[i].begin());
                if (max >= result) {
                    result = max;
                }
            }

            return result;
        }

        // Overload that finds max value in vector
        template <typename T>
        static T find_max_value(const std::vector<T>& vec)
        {
            if (vec.empty()) {
                std::cerr << "Error: Input empty" << std::endl;
                return 0;
            }

            return static_cast<T>(*std::max_element(vec.begin(), vec.end()));
        }

        // Method that finds and store maximun value in 2d matrix
        template <typename T>
        static T find_min_value(const Mat2d<T>& mat)
        {
            if (mat.empty()) {
                std::cerr << "Error: Input empty" << std::endl;
                return 0;
            }

            T result = 0;

            for (size_t i = 0; i < mat.size(); i++) {
                T min = *std::min_element(mat[i].begin(), mat[i].begin());
                if (min <= result) {
                    result = min;
                }
            }

            return result;
        }

        // Overload that finds max value in vector
        template <typename T>
        static T find_min_value(const std::vector<T>& vec)
        {
            if (vec.empty()) {
                std::cerr << "Error: Input empty" << std::endl;
                return 0;
            }

            return static_cast<T>(*std::min_element(vec.begin(), vec.end()));
        }
        
        // Method that returns rows and columns size of 2d matrix
        template <typename T>
        static std::pair<int, int> get_shape(const Mat2d<T>& mat)
        {
            int rows = mat.size();
            int cols = (rows > 0) ? mat[0].size() : 0;
            return std::make_pair(rows, cols);
        }

        // Further methods to be implemented
    };

    // Class that contains all methods that are needed for Data Analysis
    class DataAnalysis 
    {
    public:
        // Method to parse CSV line with quoted fields and blank spaces
        static std::vector<std::string> parse_csv_line(const std::string& line)
        {
            std::vector<std::string> row;

            if (!line.empty()) {
                std::stringstream ss(line);
                std::string cell;

                // Tokenize line by comma
                while (std::getline(ss, cell, ',')) {
                    // Check if cell starts with a quote
                    if (cell.front() == '"') {
                        std::string quotedCell;
                        quotedCell += cell;

                        // Keep reading until we find the closing quote
                        while (ss && cell.back() != '"') {
                            std::getline(ss, cell, ',');
                            quotedCell += "," + cell;
                        }
                        // Fill out row vector with parsed cells with quotes removed
                        quotedCell = quotedCell.substr(1, quotedCell.size() - 2);
                        row.push_back(quotedCell);
                    } 
                    else if (cell.find(' ') == std::string::npos) {
                        // Fill out row vector with parsed cells
                        // Using istringstream for cells with no space works better
                        std::istringstream iss(cell);
                        std::string val;
                        iss >> val;
                        row.push_back(val);
                    }
                    else {
                        row.push_back(cell);
                    }
                }
                // Return filled out row
                return row;
            }
            // Display error and return empty row
            std::cerr << "Error: Line provided is empty" << std::endl;
            return row;
        }
        
        // Method to read CSV files into 2d string matrix
        static Mat2d<std::string> read_csv_file(const std::string& filePath)
        {
            // Create 2d matrix to store data
            Mat2d<std::string> data;
            // Open CSV file from path provided
            std::ifstream file(filePath);
            // Check if file opened correctly
            if (!file.is_open()) {
                // Display error and return empty data if file couldn't open
                std::cerr << "Error: Coun't open file provided at '" << filePath << "'"
                        << std::endl;
                return data;
            }
            // Reach each line of the file
            std::string line;

            while (std::getline(file, line)) {
                // Parse CSV line and add it to the data matrix
                std::vector<std::string> row = parse_csv_line(line);
                data.push_back(row);
            }
            // Close file
            file.close();
            // Return data in matrix
            return data;
        }
        
        // Method to convert string to differnt data type, returns new 2d matrix with passed type
        template<typename T> 
        static Mat2d<T> matrix_converter(const Mat2d<std::string>& stringMatrix)
        {
            // Create 2d matrix of new typing
            Mat2d<T> convertedMatrix;

            if (!stringMatrix.empty()) {
                // Loop through all elements changing their typing
                for (const auto& row : stringMatrix) {
                    std::vector<T> convertedRow;

                    for (const auto& element : row) {
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
            else {
                // Display error to inform data is empty
                std::cerr << "Error: Data matrix is empty!" << std::endl;
                return convertedMatrix;
            }
        }
        
        // Converter overload that converts matrix other than string to different data type
        template <typename T, typename R>
        static Mat2d<R> matrix_converter(const Mat2d<T>& dataMatrix)
        {
            if (dataMatrix.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return Mat2d<R>();
            }

            Mat2d<R> result(dataMatrix.size(), std::vector<R>(dataMatrix[0].size()));

            for (size_t i = 0; i < result.size(); i++) {
                for (size_t j = 0; j < result[i].size(); j++) {
                    result[i][j] = static_cast<R>(dataMatrix[i][j]);
                }
            }

            return result;
        }

        // Method to format data matrix by removing rows and or columns for better data
        // analysis Template for removing single row or column
        template <typename T> 
        static void matrix_formatter(Mat2d<T>& dataMatrix, Formatter f, const int& toBeRemoved)
        {
            if (!dataMatrix.empty()) {
                if (f == COLUMN) {
                    if (toBeRemoved >= 0) {
                        // Iterate over each row and erase the element at colIndex
                        for (auto& row : dataMatrix) {
                            if (toBeRemoved < row.size()) {
                                row.erase(row.begin() + toBeRemoved);
                                return;
                            }
                        }
                        // Display error to inform invalid index to remove
                        std::cerr << "Error: Invalid index of column to remove"
                                << std::endl;
                    }
                    // Display error to inform invalid index to remove
                    std::cerr << "Error: Invalid index of column to remove"
                            << std::endl;
                } else if (f == ROW) {
                    if (toBeRemoved >= 0 && toBeRemoved <= dataMatrix.size()) {
                        // Delete selected row from matrix
                        dataMatrix.erase(dataMatrix.begin() + toBeRemoved);
                        return;
                    }
                    // Display error to inform invalid index to remove
                    std::cerr << "Error: Invalid index of row to remove" << std::endl;
                } else {
                    // Display error to inform wrong formatter enum was passed
                    std::cerr << "Error: Wrong formatter passed (2nd Paramater)"
                            << std::endl;
                }
            }
            // Display error to inform data is empty
            std::cerr << "Error: Data Matrix is empty" << std::endl;
        }
        
        // Overload to remove multiple row or columns
        // rocToRemove = rows or columns to remove, vector will hold all rows or columns to remove
        template <typename T> 
        static void matrix_formatter(Mat2d<T>& dataMatrix, Formatter f, const std::vector<int>& rocToRemove)
        {
            if (!dataMatrix.empty()) {
                if (f == COLUMN) {
                    if (!rocToRemove.empty()) {
                        // Iterate over each row and erase the element at colIndexexes
                        for (int row = 0; row < dataMatrix.size(); row++) {
                            for (auto col = rocToRemove.rbegin(); col != rocToRemove.rend(); ++col) {
                                size_t colIndex = *col;
                                if (colIndex < dataMatrix[row].size()) {
                                    dataMatrix[row].erase(dataMatrix[row].begin() +
                                                        colIndex);
                                }
                            }
                        }
                        return;
                    }
                    // Display error to inform position data is empty
                    std::cerr << "Error: No rows or columns to remove were found" << std::endl;
                } 
                else if (f == ROW) {
                    if (!rocToRemove.empty()) {
                        // Iterate through declared rows to be removed from matrix
                        for (auto item = rocToRemove.rbegin(); item != rocToRemove.rend(); ++item) {
                            size_t rowIndex = *item;
                            if (rowIndex < dataMatrix.size()) {
                                dataMatrix.erase(dataMatrix.begin() + rowIndex);
                            }
                        }
                        return;
                    }
                    // Display error to inform position data is empty
                    std::cerr << "Error: No rows or columns to remove were found" << std::endl;
                } 
                else {
                    // Display error to inform wrong formatter enum was passed
                    std::cerr << "Error: Wrong formatter passed (2nd Paramater)" << std::endl;
                }
            }
            // Display error to inform data is empty
            std::cerr << "Error: Data Matrix is empty" << std::endl;
        }
        
        // Overload to remove single or multiple row and columns
        // racToRemove = rows and columns to remove, outside vector will hold all rows and inside vector all columns to remove
        template <typename T> 
        static void matrix_formatter(Mat2d<T>& dataMatrix, Formatter f, const Mat2d<int>& racToRemove)
        {
            if (!dataMatrix.empty()) {
                if (f == ROWANDCOLUMN) {
                    if (!racToRemove.empty()) {
                        matrix_formatter(dataMatrix, ROW, racToRemove[0]);
                        matrix_formatter(dataMatrix, COLUMN, racToRemove[1]);
                        return;
                    }
                    // Display error if no position data  was found
                    std::cerr << "Error: No rows and columns to remove" << std::endl;
                }
                // Display error to inform wrong formatter enum was passed
                std::cerr << "Error: Wrong formatter passed (2nd Paramater)" << std::endl;
            }
            std::cerr << "Error: Data is empty" << std::endl;
        }
        
        // Overload to add single row or column
        template <typename T> 
        static void matrix_formatter(Mat2d<T>& dataMatrix, Formatter f, const int& indexToAdd, const std::vector<T>& dataToAdd)
        {
            if (!dataMatrix.empty()) {
                if (f == COLUMN) {
                    // Check if indexToAdd is valid and within bounds
                    if (indexToAdd >= 0 && indexToAdd <= dataMatrix[0].size()) {
                        for (int i = 0; i < dataToAdd.size(); i++) {
                            // Insert dataToAdd[i] at selected index in each row
                            dataMatrix[i].insert(dataMatrix[i].begin() + indexToAdd,
                                                dataToAdd[i]);
                        }
                        return;
                    }
                    std::cerr << "Error: Invalid index to add for inserting column." << std::endl;
                } 
                else if (f == ROW) {
                    // Check if indexToAdd is valid and within bounds
                    if (indexToAdd >= 0 && indexToAdd <= dataMatrix.size()) {
                        dataMatrix.insert(dataMatrix.begin() + indexToAdd, dataToAdd);
                        return;
                    }
                    std::cerr << "Error: Invalid index to add for inserting row." << std::endl;
                } 
                else {
                    // Display error to inform formatter enum error no data was
                    // found
                    std::cerr << "Error: Wrong formatter passed (2nd Paramater)" << std::endl;
                }
            }
            // Display error if no data was found
            std::cerr << "Error: Data Matrix is empty" << std::endl;
        }
        
        // Overload to add multiple rows or columns
        template <typename T> 
        static void matrix_formatter(Mat2d<T>& dataMatrix, Formatter f, const std::vector<int>& indexesToAdd, const Mat2d<T>& dataToAdd)
        {
            if (!dataMatrix.empty()) {
                if (f == COLUMN) {
                    // Check if indexesToAdd is not empty
                    if (!indexesToAdd.empty()) {
                        for (size_t i = 0; i < dataToAdd.size(); i++) {
                            for (size_t j = 0; j < dataToAdd[i].size(); j++) {
                                for (size_t k = 0; k < indexesToAdd.size(); k++) {
                                    if (indexesToAdd[k] < dataMatrix[j].size()) {
                                        dataMatrix[j].insert(dataMatrix[j].begin() + indexesToAdd[k],
                                                            dataToAdd[i][j]);
                                    }
                                }
                            }
                        }
                        return;
                    }
                    std::cerr << "Error: Indexes to add column are missing." << std::endl;
                    return;
                } 
                else if (f == ROW) {
                    // Check if indexesToAdd is not empty
                    if (!indexesToAdd.empty()) {
                        for (size_t i = 0; i < dataToAdd.size(); i++)
                        {
                            if (indexesToAdd[i] < dataMatrix.size() && i < dataToAdd.size())
                            {
                                dataMatrix.insert(dataMatrix.begin() + indexesToAdd[i], dataToAdd[i]);
                            }                     
                        }                
                        return;
                    }
                    std::cerr << "Error: Invalid index to add for inserting row." << std::endl;
                    return;
                } 
                else {
                    // Display error to inform formatter enum error no data was
                    // found
                    std::cerr << "Error: Wrong formatter passed (2nd Paramater)" << std::endl;
                    return;
                }
            }
            // Display error if no data was found
            std::cerr << "Error: Data Matrix is empty" << std::endl;
        }
        
        // Overload to add single or multiple rows and columns 
        // Indexes to add matrix, first row will hold all row indexes, second will hold all column indexes                                                           
        template <typename T> 
        static void matrix_formatter(Mat2d<T>& dataMatrix, Formatter f, const Mat2d<int>& indexesToAdd, const Mat2d<T>& dataToAdd)
        {
            if (!dataMatrix.empty()) {
                if (f == ROWANDCOLUMN) {
                    if (!dataToAdd.empty()) {
                        // Still need to figure out how to separate between row and column data
                        // in dataToAdd matrix
                        matrix_formatter(dataMatrix, ROW, indexesToAdd[0], dataToAdd);
                        matrix_formatter(dataMatrix, COLUMN, indexesToAdd[1], dataToAdd);
                        return;
                    }
                    // Display error if no position data  was found
                    std::cerr << "Error: No rows and columns to remove" << std::endl;
                    return;
                }
                // Display error to inform wrong formatter enum was passed
                std::cerr << "Error: Wrong formatter passed (2nd Paramater)" << std::endl;
                return;
            }
            std::cerr << "Error: Data is empty" << std::endl;
        }
        
        // Method to search data matrix for first appearance of desired element, return first position found
        template <typename T> 
        static std::vector<int> find(const Mat2d<T>& dataMatrix, const T& desiredElement)
        {
            // Creat position vector
            std::vector<int> pos{0, 0};

            if (!dataMatrix.empty()) {
                // Loop through matrix to check for desired element
                for (size_t row = 0; row < dataMatrix.size(); row++) {
                    // std::cout << row << std::endl;
                    for (size_t col = 0; col < dataMatrix[row].size(); col++) {
                        // std::cout << col << std::endl;
                        if (dataMatrix[row][col] == desiredElement) {
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
        }
        
        // Method to search data matrix by position, return element at requqested position on 2d Matrix
        template <typename T> 
        static T find_by_pos(const Mat2d<T>& dataMatrix, std::vector<int>& pos)
        {
            T element;

            if (!dataMatrix.empty()) {
                element = dataMatrix[pos[0]][pos[1]];
                return element;
            }
            // Return empty element
            std::cerr << "Error: Data Matrix is empty" << std::endl;
            return element;
        }
        
        // Method to search data matrix for all appearances of desired element, return
        // vector of positions Outside vecotor holds all rows indexes element was found,
        // and inside vector holds all columns indexes element was found
        template <typename T> 
        static Mat2d<int> find_all(const Mat2d<T>& dataMatrix, const T& desiredElemet)
        {
            // Creat position vector
            Mat2d<int> pos;

            if (!dataMatrix.empty()) {
                // Loop through matrix to check for desired element
                for (int row = 0; row < dataMatrix.size(); row++) {
                    for (int col = 0; col < dataMatrix[row].size(); col++) {
                        if (dataMatrix[row][col] == desiredElemet) {
                            std::vector<int> currentPos{row, col};
                            pos.push_back(currentPos);
                        }
                    }
                }
                if (pos.empty()) {
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
        }
        
        // Method to display all elemenst of data matrix
        template <typename T> 
        static void display_all(const Mat2d<T>& dataMatrix)
        {
            if (!dataMatrix.empty()) {
                for (auto& row : dataMatrix) {
                    for (auto& cell : row) {
                        std::cout << " " << std::to_string(cell);
                    }
                    std::cout << std::endl;
                }
                return;
            }
            // Display error that informs data matrix is empty
            std::cerr << "Error: Data Matrix is empty" << std::endl;
        }
        
        // Method to display all elements in given rows
        template <typename T> 
        static void display_rows(const Mat2d<T>& dataMatrix, const std::vector<int>& rowsToDisplay)
        {
            if (!dataMatrix.empty() && !rowsToDisplay.empty()) {
                for (size_t row = 0; row < rowsToDisplay.size(); row++) {
                    std::cout << rowsToDisplay[row] << "-";
                    if (rowsToDisplay[row] < dataMatrix.size()) {
                        for (size_t col = 0; col < dataMatrix[row].size(); col++) {
                            std::cout << " " << dataMatrix[row][col];
                        }
                        std::cout << std::endl;
                    }
                }
                // Display error that informs one or more of rows to display is
                // invalid std::cerr << "Error: One or more rows to display is
                // invalid" << std::endl;
                return;
            }
            // Display error that informs data matrix is empty
            std::cerr << "Error: Data Matrix is empty" << std::endl;
        }
        
        // Method to display all elements in given columns
        template <typename T> 
        static void display_columns(const Mat2d<T>& dataMatrix, const std::vector<int>& colsToDisplay)
        {
            if (!dataMatrix.empty() && !colsToDisplay.empty()) {
                for (size_t row = 0; row < dataMatrix.size(); row++) {
                    if (colsToDisplay[row] < dataMatrix.size()) {
                        std::cout << row << "-";
                        for (auto& col : colsToDisplay) {
                            std::cout << " " << dataMatrix[row][col];
                        }
                        std::cout << std::endl;
                    }
                }
                // Display error that informs that one or more columns to display
                // are invalid std::cerr << "Error: One or more columns to display
                // is invalid" << std::endl;
                return;
            }
            // Display error that informs data matrix is empty
            std::cerr << "Error: Data Matrix is empty" << std::endl;
        }
        
        // Method to display first five rows method, Display first 5 rows + display_head row
        template <typename T> 
        static void display_head(const Mat2d<T>& dataMatrix, int rowsToDisplay = 5)
        {
            if (!dataMatrix.empty()) {
                if (rowsToDisplay >= 0 && rowsToDisplay < dataMatrix.size() - 1) {
                    for (int row = 0; row < rowsToDisplay + 1; row++) {
                        for (int col = 0; col < dataMatrix[row].size(); col++) {
                            std::cout << " " << dataMatrix[row][col];
                        }
                        std::cout << std::endl;
                    }
                    return;
                }
                // Display error that informs number of rows is invalid matrix is
                // empty
                std::cerr << "Error: Rows to display is invalid" << std::endl;
                return;
            }
            // Display error that informs data matrix is empty
            std::cerr << "Error: Data Matrix is empty" << std::endl;
        }
        
        // Mehtod to display last five colunms method, Display lastdisplayBfive rows
        template <typename T> 
        static void display_bottom(const Mat2d<T>& dataMatrix, int rowsToDisplay = 5)
        {
            if (!dataMatrix.empty()) {
                if (rowsToDisplay >= 0 && rowsToDisplay < dataMatrix.size() - 1) {
                    for (int row = dataMatrix.size() - rowsToDisplay;
                        row < dataMatrix.size(); row++) {
                        for (int col = 0; col < dataMatrix[row].size(); col++) {
                            std::cout << " " << dataMatrix[row][col];
                        }
                        std::cout << std::endl;
                    }
                    return;
                }
                // Display error that informs number of rows is invalid matrix is
                // empty
                std::cerr << "Error: Rows to display is invalid" << std::endl;
                return;
            }
            // Display error that informs data matrix is empty
            std::cerr << "Error: Data Matrix is empty" << std::endl;
        }
        
        // Further methods to be implemented
    }; 
    
    // Class that will hold support methods for now, and later will include all methods
    // Required for computer vision
    class ComputerVision
    {
    public:
        // Method to convert 3d matrix into 2d matrix, where each element
        // Will be a sum of R, G and B values
        static Mat2d<int16_t>* get_sum_all_pixels(const Mat3d<u_int8_t>* mat3dPtr)
        {
            if (mat3dPtr == nullptr) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: 3d matrix pointer is null" << std::endl;
                return nullptr; // Return nullptr if image data is empty
            }

            // Create a new Mat2d pointer and allocate memory for it
            Mat2d<int16_t> *mat2dPtr = new Mat2d<int16_t>(mat3dPtr->size(), std::vector<int16_t>((*mat3dPtr)[0].size()));

            for (size_t i = 0; i < mat2dPtr->size(); i++) {
                for (size_t j = 0; j < (*mat2dPtr)[i].size(); j++) {
                    auto v = (*mat3dPtr)[i][j];
                    int16_t e = NumPP::get_sum_of_vector<u_int8_t, int16_t>(v);
                    (*mat2dPtr)[i][j] = e;
                }
            }

            return mat2dPtr;
        }

    };

    // Class that contains all methods that are needed for Neural Network usage
    class NeuralNetwork
    {
    private:
        /* 
        Mat2d<double>* m_a1; // Activation output for first hidden layer
        Mat2d<double>* m_a2; // Activation output for second hidden layer
        Mat2d<double>* m_a3; // Activation output for output layer
        Mat2d<double>* m_w1; // Weight matrix for the first hidden layer
        Mat2d<double>* m_w2; // Weight matrix for the scond hidden layer
        Mat2d<double>* m_w3; // Weight matrix for the output layer
        std::vector<double>* m_b1; // Bias vector for the first hidden layer
        std::vector<double>* m_b2; // Bias vector for the second hidden layer
        std::vector<double>* m_b3; // Bias vector for the output layer 
        */

        Mat2d<double> m_a1; // Activation output for first hidden layer
        Mat2d<double> m_a2; // Activation output for second hidden layer
        Mat2d<double> m_a3; // Activation output for output layer
        Mat2d<double> m_w1; // Weight matrix for the first hidden layer
        Mat2d<double> m_w2; // Weight matrix for the scond hidden layer
        Mat2d<double> m_w3; // Weight matrix for the output layer
        Mat2d<double> m_b1; // Bias vector for the first hidden layer
        Mat2d<double> m_b2; // Bias vector for the second hidden layer
        Mat2d<double> m_b3; // Bias vector for the output layer

        // Method to generate kernel matrix, which will be apllied in filter step
        template <typename T> 
        Mat2d<T>* gen_kernel(const int& kernel_size)
        {
            Mat2d<T>* p = new Mat2d<T>(NumPP::rand<T>(kernel_size, kernel_size));
            return p;
        } 

        // Method to return block of data, i.e, NxN matrix from larger matrix to apply filter
        // !!! Still hard coded, need to optimize !!!
        template <typename T>
        Mat2d<T> get_matrix_block(const Mat2d<T>* matPtr, const int& block_size, const std::vector<int>& center)
        {
            Mat2d<T> block(block_size, std::vector<T>(block_size)); // Create 2d matrix block to return later
            int offset = block_size - 2; // Get offset, number to go back and forward from center 
            
            block[0] = {(*matPtr)[center[0]-offset][center[1]-offset], (*matPtr)[center[0] - offset][center[1]], (*matPtr)[center[0]-offset][center[1]+offset]};
            block[1] = {(*matPtr)[center[0]][center[1]-offset], (*matPtr)[center[0]][center[1]], (*matPtr)[center[0]][center[1]+offset]};
            block[2] = {(*matPtr)[center[0]+offset][center[1]-offset], (*matPtr)[center[0]+offset][center[1]], (*matPtr)[center[0]+offset][center[1]+offset]};

            return block;
        }

        // Aplly filter to input, remove unwanted chunks, makes further steps more efficient
        // !!! STILL NEED TO CHECK IF IT'S WORKING 100% CORRECTLY !!!
        Mat2d<int16_t>* pre_process_input(const Mat2d<int16_t>* dataPtr, const int& kernel_size)
        {
            int offset = kernel_size - 2; // Get offset, number to go back and forward from center    
            Mat2d<int16_t>* fMat = new Mat2d<int16_t>(dataPtr->size()-(offset*2), std::vector<int16_t>((*dataPtr)[0].size()-(offset*2))); // Create pointer to store final data
            Mat2d<int16_t>* block_mat = new Mat2d<int16_t>(kernel_size, std::vector<int16_t>(kernel_size)); // Create memory to store NxN block matrix
            Mat2d<int16_t>* kMat = gen_kernel<int16_t>(kernel_size); // Generate kernel matrix with passed size
            DataAnalysis::display_all(*kMat);
            std::vector<int> center(2);

            // Loop through image matrix, centering kernel matrix with block to be analysed
            for (int center_x = offset; center_x < dataPtr->size() - offset; center_x++) {  
                //std::vector<int16_t> rowData;
                for (int center_y = offset; center_y < (*dataPtr)[center_x].size() - offset; center_y++) {
                    center = {center_x, center_y};      
                    *block_mat = get_matrix_block(dataPtr, kernel_size, center); // Get NxN block of image matrix
                    (*fMat)[center_x-1][center_y-1] = NumPP::sum_mat_mul_matching_elements(block_mat, kMat); // Multiply block by kernel matrix
                    //rowData.push_back(r);                
                }
                //fMat->push_back(rowData);
            }

            delete kMat, block_mat;
            return fMat;
        }
    public:
        NeuralNetwork(int input_size, int hidden_size1, int hidden_size2, int output_size) 
        {
           // Initialize weights and biases randomly
           init_weights(input_size, hidden_size1, hidden_size2, output_size);
        }

        // Method to initialize weights and biases randomly
        void init_weights(int& input_size, int& hidden_size1, int& hidden_size2, int& output_size)
        {
            // Initialize weights with random values
            m_w1 = NumPP::rand<double>(input_size, hidden_size1);
            m_w2 = NumPP::rand<double>(hidden_size1, hidden_size2);
            m_w3 = NumPP::rand<double>(hidden_size2, output_size);

            // IUnitialize biases with zeros
            m_b1 = NumPP::zeros<double>(1, hidden_size1);
            m_b2 = NumPP::zeros<double>(1, hidden_size2);
            m_b3 = NumPP::zeros<double>(1, output_size);
        }

        // Method to perform forward propagation and compute output
        template <typename T>
        Mat2d<double> forward(const Mat2d<T>& x)
        { 
            // Compute output of first hidden layer
            Mat2d<double> z1 = NumPP::add(NumPP::dot<double>(x, m_w1), m_b1[0]); // Linear transformation for the first hidden layer
            m_a1 = NumPP::tanh(z1); // Activation output for the first hidden layer

            // Compute output of second hidden layer
            Mat2d<double> z2 = NumPP::add(NumPP::dot<double>(m_a1, m_w2), m_b2[0]); // Linear transformation for the second hidden layer
            m_a2 = NumPP::tanh(z2); // Activation output for the second hidden layer

            // Compute outpute layer
            Mat2d<double> z3 = NumPP::add(NumPP::dot<double>(m_a2, m_w3), m_b3[0]); // Linear transformation for the output layer
            m_a3 = NumPP::tanh(z3); // Activation output for the output layer

            return m_a3; // Return output result
        }

        // Method to uptade weights and biases
        void update_parameters(const Mat2d<double>& dw1, const Mat2d<double>& dw2, const Mat2d<double>& dw3,
                                const Mat2d<double>& db1, const Mat2d<double>& db2, const Mat2d<double>& db3, 
                                const double& learning_rate)
        {
            m_w1 = NumPP::subtract(m_w1, NumPP::scalar_mat_mul(dw1, learning_rate)); // Update weights of the first hidden layer
            m_b1 = NumPP::subtract(m_b1, NumPP::scalar_mat_mul(db1, learning_rate)); // Update biases of the first hidden layer
            m_w2 = NumPP::subtract(m_w2, NumPP::scalar_mat_mul(dw2, learning_rate)); // Update weights of the first hidden layer
            m_b2 = NumPP::subtract(m_b2, NumPP::scalar_mat_mul(db2, learning_rate)); // Update biases of the first hidden layer
            m_w3 = NumPP::subtract(m_w3, NumPP::scalar_mat_mul(dw3, learning_rate)); // Update weights of the first hidden layer
            m_b3 = NumPP::subtract(m_b3, NumPP::scalar_mat_mul(db3, learning_rate)); // Update biases of the first hidden layer 
        }

        // Method to perform backward propagation and update weights
        template <typename T>
        void backward(const Mat2d<T>& x, const Mat2d<T>& y, const double& learning_rate)
        {
            double n = x.size(); // Number of training example
            // Gradient of the loss with respect to the output layer;
            Mat2d<double> dz3 = NumPP::subtract(m_a3, y);
            // Gradient of the loss with respect to the weights of the output layer
            Mat2d<double> dw3 = NumPP::scalar_mat_mul(NumPP::dot<double>(NumPP::transpose(m_a2), dz3), (1/n)); 
            // Gradient of the loss with respect to the biases of the output layer
            Mat2d<double> db3(1, std::vector<double>(dz3[0].size()));
            db3[0] = NumPP::scalar_vec_mul(NumPP::sum(dz3, 1), (1/n));

            // Create variables to hold sub operations
            Mat2d<double> a = NumPP::dot(dz3, NumPP::transpose(m_w3));
            Mat2d<double> b = NumPP::scalar_mat_sub<double>(1.0, NumPP::power(m_a2, 2));
            // Gradient of the loss with respect to the output of the second hidden layer 
            Mat2d<double> dz2 = NumPP::mat_mul_matching_elements(a, b); 
            // Gradient of the loss with respect to the weights of the second hidden layer
            Mat2d<double> dw2 = NumPP::scalar_mat_mul(NumPP::dot(NumPP::transpose(m_a1), dz2), (1/n));
            // Gradient of the loss with respect to the biases of the second hidden layer
            Mat2d<double> db2(1, std::vector<double>(dz2[0].size()));
            db2[0] = NumPP::scalar_vec_mul(NumPP::sum(dz2, 1), (1/n));

            // Override previously created pointers with newly appointed ones
            a = NumPP::dot(dz2, NumPP::transpose(m_w2));
            b = NumPP::scalar_mat_sub<double>(1.0, NumPP::power(m_a1, 2));
            // Gradient of the loss with respect to the output of the first hidden layer
            Mat2d<double> dz1 = NumPP::mat_mul_matching_elements(a, b);
            // Gradient of the loss with respect to the weights of the first hidden layer
            Mat2d<double> dw1 = NumPP::scalar_mat_mul(NumPP::dot(NumPP::transpose(x), dz1), (1/n));
            // Gradient of the loss with respect to the biases of the first hidden layer
            Mat2d<double> db1(1, std::vector<double>(dz1[0].size()));
            db1[0] = NumPP::scalar_vec_mul(NumPP::sum(dz1, 1), (1/n));

            update_parameters(dw1, dw2, dw3, db1, db2, db3, learning_rate); // Update weights and biases
            
        }

        // Further methods to be implemented
    };

    // Class that transforms open cv data structures into mine for later analysis
    // This is temporary, I do not plan to use open cv in public release, this is for learning and testing purposes
    class OpencvIntegration
    {
    private:
        // Get sum of a open cv Vec3b data type, returns the sum in declared data type
        template <typename T>
        static T get_sum_of_vector(const cv::Vec3b& vec)
        {
            T result; // Store result in desired data type

            for (size_t i = 0; i < vec.channels; i++) {
                result += static_cast<T>(vec[i]);
            }

            return result;

        }
    public:
        // Get open cv image matrix and convert it to our 3d matrix data structure
        // Default format is Blue, Green, Red
        static Mat3d<u_int8_t>* convert_image(const cv::Mat* image) 
        {
            if (!image->data) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Image data empty" << std::endl;
                return nullptr; // Return nullptr if image data is empty
            }

            // Create a new Mat3d pointer and allocate memory for it
            Mat3d<u_int8_t> *nMatPtr = new Mat3d<u_int8_t>(image->rows, Mat2d<uint8_t>(image->cols, std::vector<uint8_t>(3)));
            // Create pointer to hold open cv pixel values
            cv::Vec3b* cvPixelPtr = new cv::Vec3b(3);
            // Iterate over each pixel in the given image
            for (int i = 0; i < image->rows; i++) {
                for (int j = 0; j < image->cols; j++) {
                    // Assing current pixel data to pointer
                    (*cvPixelPtr) = image->at<cv::Vec3b>(i, j);
                    // Access the corresponding pixel in the new matrix and copy pixel values
                    (*nMatPtr)[i][j] = {(*cvPixelPtr)[0], (*cvPixelPtr)[1], (*cvPixelPtr)[2]};
                }
            }

            delete cvPixelPtr;
            // Return the pointer to the final matrix
            return nMatPtr;
        }

        // Convert open cv matrix, to 2d matrix, where each element will be the sum
        // Of Blue, Green and Red values
        static Mat2d<int16_t>* get_sum_pixels(const cv::Mat* image)
        {
            if (!image->data) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Image data empty" << std::endl;
                return nullptr; // Return nullptr if image data is empty
            }

            // Create a new Mat2d pointer and allocate memory for it
            Mat2d<int16_t> *mat2dPtr = new Mat2d<int16_t>(image->rows, std::vector<int16_t>(image->cols));

            for (size_t i = 0; i < mat2dPtr->size(); i++) {
                for (size_t j = 0; j < (*mat2dPtr)[i].size(); j++) {
                    auto v = image->at<cv::Vec3b>(i, j);
                    int16_t e = get_sum_of_vector<int16_t>(v);
                    (*mat2dPtr)[i][j] = e;
                }
            }

            return mat2dPtr;
        }
        // Further methods to be implemented
    };
    
    // Further classes to be implemented
}

