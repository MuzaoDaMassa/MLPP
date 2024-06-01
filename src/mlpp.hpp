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
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <any>
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
#pragma region New Data Structure declarations
    // Declaring 2d Matrix template, which is a vector that holds other vectors
    // Ex: Mat2d[0] = vector stored in at index 0, Mat2d[0][0] = single value stored at index (0,0)
    // Essentially Mat2d[x][y] will return value stored in row x, column y
    template <typename T> using Mat2d = std::vector<std::vector<T>>;
    // Declare 3d matrix template which is a vector that holds 2d matrices
    // Ex: Mat3d[0] = 2d matrix stored at index 0, Mat3d[0][0] = vector stored at index 0 in stored 2d matrix,
    // Mat3d[0][0][0] = single value stored at index (0,0,0)
    // Essentially Mat3d[x][y][z] will return value stored in row x, column y, depth z
    template <typename T> using Mat3d = std::vector<Mat2d<T>>;  
    // Declare 4d matrix template where each element is a 3d matrix
    // Ex: Mat4d[0] = 3d matrix stored at index 0, Mat4d[0][0] = 2d matrix stored at index (0,0),
    // Mat4d[0][0][0] = vector stored at index (0,0,0), Mat4d[0][0][0][0] = single value stored at index (0,0,0,0)
    // Essentially Mat4d[a][x][y][z] = will return values stored in 3d matrix a, in row x, column y, depth z
    template <typename T> using Mat4d = std::vector<Mat3d<T>>; 
#pragma endregion

#pragma region Enums
    // Formatter utility enum to help with method overload
    enum Formatter { ROW, COLUMN, ROWANDCOLUMN };
    // Activation functions enum for neural networks
    enum Activation { RELU, TANH, SIGMOID, SOFTMAX };
    // Padding enum for convolution layers
    enum Padding { SAME, VALID };
#pragma endregion

#pragma region NumPP
    // Class that contains all methods that are needed for numeric computation
    class NumPP
    {
    public:
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

        // Overload of add method that adds two vectors of same shape
        template <typename T>
        static std::vector<T> add(const std::vector<T>& a, const std::vector<T>& b)
        {
            if (a.empty() || b.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return std::vector<T>();
            }

            if (a.size() != b.size()) {
                std::cerr << "Error: Incompatible shapes" << std::endl;
                return std::vector<T>();
            }

            std::vector<T> output_vector(a.size());

            for (size_t i = 0; i < output_vector.size(); i++) {
                output_vector[i] = a[i] + b[i];
            }

            return output_vector;
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

        // Overload of dot method that multiplies vector with 2d matrix and return a vector
        // I.E, we treat vector as 2d matrix of size (0,R) and multiply with matrix of size (R,C)
        // Returning a matrix of size (0,C), which is treated as vector to return
        template <typename T> 
        static std::vector<T> dot(const std::vector<T>& a, const Mat2d<T>& b)
        {
            if (a.empty() || b.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return std::vector<T>();
            }

            if (a.size() != b.size()) {
                std::cerr << "Error: Incompatible shapes" << std::endl;
                return std::vector<T>();
            }

            std::vector<T> output_vector(b[0].size()); // Create output vector with b matrix columns size
            T r; // Create variable to store temporary results

            for (size_t i = 0; i < output_vector.size(); i++) {
                r = 0;
                for (size_t j = 0; j < a.size(); j++) {
                    r += a[j] * b[j][i];
                }
                output_vector[i] = r; // Fill output vector with dot result from row j and column i
            }

            return output_vector;
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
       
        // Method that returns rows and columns size of 2d matrix
        template <typename T>
        static std::pair<int, int> get_shape(const Mat2d<T>& mat)
        {
            int rows = mat.size();
            int cols = (rows > 0) ? mat[0].size() : 0;
            return std::make_pair(rows, cols);
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

        // Overload for method with uniform distribution
        template <typename T>
        static Mat2d<T> rand(const size_t& rows, const size_t& cols, const int& min, const int& max)
        {
            // Create new 2d matrix object with specified parameters to later return
            Mat2d<T> result(rows, std::vector<T>(cols));

            // Initialize random generator - Uniform distribution, generate integer values between set interval
            std::random_device rd;
            std::mt19937 generator(rd());
            std::uniform_int_distribution<int> distribution(min, max);

            // Fill matrix with random values
            for (size_t rows = 0; rows < result.size(); rows++) {
                for (size_t cols = 0; cols < result[rows].size(); cols++) {
                    result[rows][cols] = static_cast<T>(distribution(generator));
                }
            }

            return result;
        }

        // Method that applies Rectified Linear Unit function to input
        template <typename T>
        static T relu(const T& input)
        {
            if (input <= 0) {
                return 0;
            }
            return input;
        }

        // Overload method that applies Rectified Linear Unit to every value in vector
        template <typename T>
        static std::vector<T> relu(const std::vector<T>& vec)
        {
            if (vec.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return std::vector<T>();
            }

            std::vector<T> output_vector(vec.size());

            for (size_t i = 0; i < output_vector.size(); i++) {
                output_vector[i] = NumPP::relu(vec[i]);
            }

            return output_vector;
        }

        // Overload method that applies Rectified Linear Unit to every value in 3d matrix
        template <typename T>
        static Mat3d<T> relu(const Mat3d<T>& mat)
        {
            if (mat.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return Mat3d<T>();
            }

            Mat3d<T> relu_mat(mat.size(), Mat2d<T>(mat[0].size(), std::vector<T>(mat[0][0].size())));

            for (size_t i = 0; i < relu_mat.size(); i++) {
                for (size_t j = 0; j < relu_mat[i].size(); j++) {
                    for (size_t k = 0; k < relu_mat[i][j].size(); k++) {
                        relu_mat[i][j][k] = NumPP::relu(mat[i][j][k]);
                    }
                }
            }

            return relu_mat;
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

        // Method that applies sigmoid function to vector
        template <typename T> 
        static std::vector<T> sigmoid(const std::vector<T>& input_vector)
        {
            if (input_vector.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return std::vector<T>();
            }

            std::vector<T> output_vector(input_vector.size());
            const T scaling_factor = 10000.0; // Smooth values to fit c++ calculations limit
            T adjusted_value = 0.0;

            for (size_t i = 0; i < output_vector.size(); i++) {
                adjusted_value = input_vector[i];        
               
                //std::cout << adjusted_value << std::endl;
                output_vector[i] = static_cast<T>(1 / (1 + std::exp(static_cast<double>(-adjusted_value))));
                adjusted_value = 0;
            }

            return output_vector;
        }

        // Method that applies softmax function to vector
        // !!! Still need to work on method a little more, make sure everything is working properly !!!
        template <typename T> 
        static std::vector<T> softmax(const std::vector<T>& input_vector)
        {
            if (input_vector.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return std::vector<T>();
            }

            std::cout << input_vector[0] << ", " << input_vector[1] << ", " << input_vector[2] << std::endl;
            std::vector<T> exp_values(input_vector.size()); // Create vector to store computed exponential values

            // Find the maximum value in the input to improve numerical stability
            T max_input = static_cast<double>(NumPP::find_max_value(input_vector));
            // Determine a scaling factor to ensure values fall within a manageable range
            const T scaling_factor = 100000.0;
            // Create vector to hold adjusted values
            std::vector<T> adjusted_values(input_vector.size());

            // Subtract max input to current input value and apply scaling
            for (size_t i = 0; i < adjusted_values.size(); i++)
            {   
                // Get adjusted value for current input value
                adjusted_values[i] = (input_vector[i] - max_input) / scaling_factor;
                std::cout << "Adjusted Value " << i << ": " << adjusted_values[i] << std::endl;
                // Apply exponential function to adjusted value
                exp_values[i] = std::exp(adjusted_values[i]); 
                std::cout << "Exp Value " << i << ": " << exp_values[i] << std::endl;
            }         
            
            // Compute the sum of exponentials
            T sum_exp = std::accumulate(exp_values.begin(), exp_values.end(), static_cast<T>(0));
            std::cout << "Sum " << sum_exp << std::endl;

            // Normalize exponentials to get probabilities
            for (size_t i = 0; i < exp_values.size(); i++)
            {   
                exp_values[i] /= sum_exp;
            }
            
            return exp_values;
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
        
        // Overload that receives matrices of differente types and returns specified type
        template <typename R, typename A, typename B> 
        static R sum_mat_mul_matching_elements(const Mat2d<A>& a, const Mat2d<B>& b)
        {
            if (a.empty() || b.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return 0; // Return 0
            }

            R r = 0; // Create variable to store results
            // Iterate through every element in a, and multiply by matching elemnt in b
            for (size_t i = 0; i < a.size(); i++) {
                for (size_t j = 0; j < a[i].size(); j++) {
                    r += static_cast<R>(a[i][j] * b[i][j]);
                }
            }

            return r; 
        } 

        // Method that apply tanh function to every element in vector
        template <typename T>
        static std::vector<T> tanh(const std::vector<T>& vec)
        {
            if (vec.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return std::vector<T>();
            }

            std::vector<T> output_vector(vec.size());

            for (size_t i = 0; i < output_vector.size(); i++) {
                output_vector[i] = static_cast<T>(std::tanh(vec[i]));
            }

            return output_vector;
        }

        // Overload of tanh ethod that apply tanh function to every element in 2d matrix
        template <typename T>
        static Mat2d<T> tanh(const Mat2d<T>& mat)
        {
            if (mat.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<T>(); // Return null pointer        
            }

            Mat2d<T> tanhMat(mat.size(), std::vector<T>(mat[0].size()));

            for (size_t i = 0; i < tanhMat.size(); i++) {
                for (size_t j = 0; j < tanhMat[i].size(); j++) {
                    tanhMat[i][j] = static_cast<T>(std::tanh(mat[i][j]));
                }
            }
            return tanhMat;
        }  

        // Overload of tanh method to deal with 3d matrices
        template <typename T>
        static Mat3d<T> tanh(const Mat3d<T>& mat)
        {
            if (mat.empty()) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Input empty" << std::endl;
                return Mat3d<T>(); // Return null pointer        
            }

            Mat3d<T> tanhMat(mat.size(), Mat2d<T>(mat[0].size(), std::vector<T>(mat[0][0].size())));

            for (size_t i = 0; i < tanhMat.size(); i++) {
                for (size_t j = 0; j < tanhMat[i].size(); j++) {
                    for (size_t k = 0; k < tanhMat[i][j].size(); k++) {
                        tanhMat[i][j][k] = static_cast<T>(std::tanh(mat[i][j][k]));
                    }
                }
            }
            return tanhMat;
        }
   
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

        // Further methods to be implemented
    };

#pragma endregion

#pragma region DataAnalysis
    // Class that contains all methods that are needed for Data Analysis
    class DataAnalysis 
    {
    public:
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
        
        // Method to search data matrix for all appearances of desired element, return
        // vector of positions Outside vecotor holds all rows indexes element was found,
        // and inside vector holds all columns indexes element was found
        template <typename T> 
        static Mat2d<int> find_all(const Mat2d<T>& dataMatrix, const T& desiredElement)
        {
            // Creat position vector
            Mat2d<int> pos;

            if (!dataMatrix.empty()) {
                // Loop through matrix to check for desired element
                for (int row = 0; row < dataMatrix.size(); row++) {
                    for (int col = 0; col < dataMatrix[row].size(); col++) {
                        if (dataMatrix[row][col] == desiredElement) {
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
        
        // Overload of find_all that returns number of times element was found, not their position (Mat2d)
        template <typename T>
        static int find_all(const T& desiredElement, const Mat2d<T>& dataMatrix)
        {
            if (dataMatrix.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return 0;
            }

            int counter = 0;

            for (size_t i = 0; i < dataMatrix.size(); i++) {
                for (size_t j = 0; j < dataMatrix[i].size(); j++) {
                    if (dataMatrix[i][j] == desiredElement) {
                        counter++;
                    }
                }
            }

            return counter;
        }

        // Overload of find_all that returns number of times element was found, not their position (Mat3d)
        template <typename T>
        static int find_all(const T& desiredElement, const Mat3d<T>& dataMatrix)
        {
            if (dataMatrix.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return 0;
            }

            int counter = 0;

            for (size_t d = 0; d < dataMatrix[0][0].size(); d++) {
                for (size_t i = 0; i < dataMatrix.size(); i++) {
                    for (size_t j = 0; j < dataMatrix[i].size(); j++) {
                        if (dataMatrix[i][j][d] == desiredElement) {
                            counter++;
                        }
                    }
                }
            }

            return counter;
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
        
        // Further methods to be implemented
    }; 

#pragma endregion

#pragma region ComputerVision
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

        // Overload to receive double 3d matrix
        static Mat2d<double>* get_sum_all_pixels(const Mat3d<double>* mat3dPtr)
        {
            if (mat3dPtr == nullptr) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: 3d matrix pointer is null" << std::endl;
                return nullptr; // Return nullptr if image data is empty
            }

            // Create a new Mat2d pointer and allocate memory for it
            Mat2d<double>* mat2dPtr = new Mat2d<double>(mat3dPtr->size(), std::vector<double>((*mat3dPtr)[0].size()));

            for (size_t i = 0; i < mat2dPtr->size(); i++) {
                for (size_t j = 0; j < (*mat2dPtr)[i].size(); j++) {
                    auto v = (*mat3dPtr)[i][j];
                    double e = NumPP::get_sum_of_vector(v);
                    (*mat2dPtr)[i][j] = e;
                }
            }

            return mat2dPtr;
        }
       
        // Method that normalize pixel values to interval [0,1]
        template <typename T>
        static Mat3d<T> normalize_pixels(const Mat3d<T>* matPtr)
        {
            if (matPtr == nullptr) {
                std::cerr << "Error: Input is null" << std::endl;
                return Mat3d<T>();
            }

            Mat3d<T> nMat(matPtr->size(), Mat2d<T>((*matPtr)[0].size(), std::vector<T>((*matPtr)[0][0].size())));

            for (size_t i = 0; i < matPtr->size(); i++) {
                for (size_t j = 0; j < (*matPtr)[i].size(); j++) {
                    for (size_t k = 0; k < (*matPtr)[i][j].size(); k++) {
                        nMat[i][j][k] = static_cast<T>((*matPtr)[i][j][k]) / 255.0;
                    }
                }
            }

            return nMat;
        }

    };

#pragma endregion

#pragma region NeuralNetwork
    // Abstract base layer class
    class LayerBase 
    {
    protected:
        const bool m_use_weights_and_biases;
    public:
        LayerBase(const bool use_weights_and_biases) : m_use_weights_and_biases(use_weights_and_biases) {}
        virtual ~LayerBase() = default;
        //virtual void forward(void* input, void* output) = 0;
        //virtual void forward(void* input, void* output, void* weights, void* bias) = 0;
        virtual void forward(void* input, void* output, std::any& weights, std::any& bias) = 0;
        bool get_w_and_b() {return m_use_weights_and_biases;} // testing only
    };

    // Basic layer methods and define layer input, output, and data types
    template <typename InputType, typename OutputType, typename DataType>
    class Layer : public LayerBase
    {
    public:
        Layer(const bool use_weights_and_biases) : LayerBase(use_weights_and_biases) {};
        virtual ~Layer() = default;
    protected:
        //void forward(void* input, void* output) override {}
        //void forward(void* input, void* output, void* weights, void* bias) override {}
        void forward(void* input, void* output, std::any& weights, std::any& bias) override {}
        virtual Mat3d<DataType> activation_process(const Mat3d<DataType>& mat, const Activation& activation_function) {return Mat3d<DataType>();}
        virtual std::vector<DataType> activation_process(const std::vector<DataType>& vec, const Activation& activation_function) {return std::vector<DataType>();}
        virtual Mat3d<DataType> conv_2d_process(const Mat3d<DataType>& mat, const Mat3d<DataType>& kernel_mats, const std::vector<DataType>& bias_vec, 
                                                const Padding& padding, const int& kernel_size, const int& number_of_filters) {return Mat3d<DataType>();}
        virtual std::vector<DataType> flatten_process(const Mat3d<DataType>& mat) {return std::vector<DataType>();}
        virtual Mat2d<DataType> get_2d_block_from_mat(const Mat3d<DataType>& mat, const int& block_size, const size_t& offset, std::pair<size_t, size_t>& output_loc,
                                                        const size_t& depth) {return Mat2d<DataType>();}
        virtual Mat3d<DataType> max_pool_process(const Mat3d<DataType>& mat, const int& size, const int& stride, const int& depth) {return Mat3d<DataType>();}
    };

    // Devired Layer Class that creates Convolutional layer
    template <typename InputType, typename OutputType, typename DataType>
    class Conv2D : public Layer<InputType,OutputType,DataType>
    {
    private:
        const int m_number_of_filters; // Member Variable to hold passed number of filters
        const int m_kernel_size; // Member Variable to hold kernel matrix size, square matrix so only 1 value needed
        const Activation m_activation_function; // Member Variable to hold passed activation function method
        const Padding m_padding; // Member Variable to hold passed padding to layer
        //Mat3d<DataType> m_filters; // Member Variable to hold all filters
        std::vector<DataType> m_bias_vector; // Member variable to holl bias vector
        Mat3d<DataType> m_feature_maps; // Member Variable to hold current created feature map
        Mat3d<DataType> m_activated_feature_map; // Member Variable to hold current activated feature map
        Mat2d<DataType> m_filter{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}; // !!! Testing only !!!

        // Method that applies activation function 
        Mat3d<DataType> activation_process(const Mat3d<DataType>& mat, const Activation& activation_function) override
        {
            if (mat.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return Mat3d<DataType>();
            }

            switch (activation_function)
            {
            case RELU:
                return NumPP::relu(mat);
            case TANH:
                return NumPP::tanh(mat);
            default:
                return Mat3d<DataType>();
            }
        }

        // Method that applies filter(kernel) to 3d matrix, will be used in Convolutional layer for neural network applications     
        Mat3d<DataType> conv_2d_process(const Mat3d<DataType>& mat, const Mat3d<DataType>& kernel_mats, const std::vector<DataType>& bias_vec, 
                                        const Padding& padding, const int& kernel_size, const int& number_of_filters) override
        {
            if (mat.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return Mat3d<DataType>();
            }
            
            size_t offset; // Create variable to store offset, determines output size

            if (padding == SAME) {
                offset = 0;
            }
            else {
                offset = kernel_size - 1;
            }

            // Create filtered matrix variable to store and return result
            Mat3d<DataType> feature_maps(mat.size()-offset, Mat2d<DataType>(mat[0].size()-offset, std::vector<DataType>(number_of_filters))); 
            std::pair<size_t, size_t> output_location; // Create pair to store output(filtered matrix) current location
            int channels = mat[0][0].size(); // Get depth of original image
            std::vector<DataType> filtered_pixel(channels); // Create vector to store convolution iteration result    

            // Loop through input matrix N times and fill feature maps based on number of filters
            for (size_t f = 0; f < number_of_filters; f++) {
                for (size_t i = 0; i < feature_maps.size(); i++) {
                    for (size_t j = 0; j < feature_maps[i].size(); j++) {
                        output_location = {i, j};
                        for (size_t k = 0; k < channels; k++) { 
                            Mat2d<DataType> block_mat = get_2d_block_from_mat(mat, kernel_size, offset, output_location, k);
                            filtered_pixel[k] = NumPP::sum_mat_mul_matching_elements<DataType>(block_mat, kernel_mats[f]);
                           /*  if (k == 0) {
                                // Blue channel
                                filtered_pixel[k] = NumPP::sum_mat_mul_matching_elements<DataType>(block_mat, kernel_mat);
                            }
                            else if (k == 1) {
                                // Green channel
                                filtered_pixel[k] = NumPP::sum_mat_mul_matching_elements<DataType>(block_mat, kernel_mat);
                            }
                            else {
                                // Red channel
                                filtered_pixel[k] = NumPP::sum_mat_mul_matching_elements<DataType>(block_mat, kernel_mat);
                            } */
                        }
                        feature_maps[i][j][f] = NumPP::get_sum_of_vector<DataType>(filtered_pixel) + bias_vec[f]; // Get sum of all channel and add bias
                    }
                }
            }

            return feature_maps;
        }
        
        // Method that returns 2d matrix block taken from 3d matrix   
        Mat2d<DataType> get_2d_block_from_mat(const Mat3d<DataType>& mat, const int& block_size, const size_t& offset, std::pair<size_t, size_t>& output_loc,
                                                const size_t& depth) override
        {
            if (mat.empty()) {
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<DataType>();
            }

            Mat2d<DataType> b_mat(block_size, std::vector<DataType>(block_size)); // Create block mat variable to store result

            for (size_t i = 0; i < b_mat.size(); i++) {
                for (size_t j = 0; j < b_mat[i].size(); j++) {   
                    std::pair<size_t, size_t> og_mat_loc = {output_loc.first+i, output_loc.second+j};
                    if (og_mat_loc.first < mat.size() - offset && og_mat_loc.second < mat[i].size() - offset) {
                        b_mat[i][j] = mat[og_mat_loc.first][og_mat_loc.second][depth];
                    }
                }
            }

            return b_mat;
        }
 
    public:
        // Class constructor to receive correct hyperparameters
        Conv2D(const int& number_of_filters, const int& kernel_size,
        const Activation& activation_function, const Padding& padding):
            Layer<InputType, OutputType, DataType>(true), 
            m_number_of_filters(number_of_filters),
            m_kernel_size(kernel_size),
            m_activation_function(activation_function),
            m_padding(padding)
        {}

        // Override of layer forward method declared in base class
        void forward(void* input, void* output, std::any& weights, std::any& bias) override 
        {
            if (input == nullptr) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            }   
            
            InputType* typedInput = static_cast<InputType*>(input);
            OutputType* typedOutput = static_cast<OutputType*>(output);
            size_t size = typedInput->size();

            if (!weights.has_value()) {
                weights = Mat3d<DataType>();
            }

            if (!bias.has_value()) {
                bias = std::vector<DataType>();
            }

            Mat3d<DataType>* weights_input = std::any_cast<Mat3d<DataType>>(&weights);       
            std::vector<DataType>* biases_input = std::any_cast<std::vector<DataType>>(&bias);

            if (weights_input->size() == 0) {
                 for (size_t i = 0; i < m_number_of_filters; i++) {
                    weights_input->push_back(m_filter);
                }
            } 

            if (biases_input->size() == 0) {
                for (size_t i = 0; i < m_number_of_filters; i++) {
                    biases_input->push_back(0);
                }
            } 
     
            // !!! Need to optimize this !!!
            if (typedOutput->size() == 0) {
                // Loop through image and apply max pooling process
                for (size_t i = 0; i < size; i++) {
                    m_feature_maps = conv_2d_process((*typedInput)[i], *weights_input, *biases_input, m_padding, m_kernel_size, m_number_of_filters); 
                    m_activated_feature_map = activation_process(m_feature_maps, m_activation_function);
                    typedOutput->push_back(m_activated_feature_map);
                }

                return;
            } 

            // Loop through image and apply max pooling process
            for (size_t i = 0; i < size; i++) {
                m_feature_maps = conv_2d_process((*typedInput)[i], *weights_input, *biases_input, m_padding, m_kernel_size, m_number_of_filters); 
                m_activated_feature_map = activation_process(m_feature_maps, m_activation_function);
                //typedOutput->push_back(m_activated_feature_map);
                (*typedOutput)[i] = m_activated_feature_map;
            }    
        }
    };

     // Derived Layer Class that creates Max Pooling layer
    template <typename InputType, typename OutputType, typename DataType>
    class MaxPooling2D : public Layer<InputType,OutputType,DataType>
    {
    private:
        const int m_size; // Member variable to store how many pixels to pool
        const int m_stride; // Member variable to store stride to traverse through image
        int m_depth; // Member variable to store image depth
        Mat3d<DataType> m_pooled_mat; // Member variable to store current pooled matrix

        // Method that returns 2d matrix block taken from 3d matrix   
        Mat2d<DataType> get_2d_block_from_mat(const Mat3d<DataType>& mat, const int& block_size, const size_t& offset, std::pair<size_t, size_t>& output_loc,
                                                const size_t& depth) override
        {
            if (mat.empty()) {
                std::cerr << "Error: Input empty" << std::endl;
                return Mat2d<DataType>();
            }

            Mat2d<DataType> b_mat(block_size, std::vector<DataType>(block_size)); // Create block mat variable to store result

            for (size_t i = 0; i < b_mat.size(); i++) {
                for (size_t j = 0; j < b_mat[i].size(); j++) {   
                    std::pair<size_t, size_t> og_mat_loc = {output_loc.first+i, output_loc.second+j};
                    if (og_mat_loc.first < mat.size() - offset && og_mat_loc.second < mat[i].size() - offset) {
                        b_mat[i][j] = mat[og_mat_loc.first][og_mat_loc.second][depth];
                    }
                }
            }

            return b_mat;
        }

        // Method that iterates through 3d matrix and reduces it by selecting max value for each iteration
        Mat3d<DataType> max_pool_process(const Mat3d<DataType>& mat, const int& size, const int& stride, const int& depth) override
        {
            if (mat.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return Mat3d<DataType>();
            }

            Mat3d<DataType> pool_mat(mat.size()/2, Mat2d<DataType>(mat[0].size()/2, std::vector<DataType>(depth)));
            std::pair<size_t, size_t> og_mat_loc = {0, 0};

            for (size_t f = 0; f < depth; f++) {
                og_mat_loc.first = 0;
                for (size_t i = 0; i < pool_mat.size(); i++) {
                    og_mat_loc.second = 0;
                    for (size_t j = 0; j < pool_mat[i].size(); j++) {
                        Mat2d<DataType> block_mat = get_2d_block_from_mat(mat, size, 0, og_mat_loc, f);
                        pool_mat[i][j][f] = NumPP::find_max_value(block_mat);
                        og_mat_loc.second += stride;
                    }
                    og_mat_loc.first += stride;
                }
            }

            return pool_mat;
        } 

    public:
        // Class constructor to receive correct hyperparameters
        MaxPooling2D(const int& size, const int& stride):
            Layer<InputType, OutputType, DataType>(false), 
            m_size(size),
            m_stride(stride)
        {}

        // Override of layer forward method declared in base class
        void forward(void* input, void* output, std::any& weights, std::any& bias) override
        {
            if (input == nullptr) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            }

            InputType* typedInput = static_cast<InputType*>(input);
            OutputType* typedOutput = static_cast<OutputType*>(output);

            m_depth = (*typedInput)[0][0][0].size();
            size_t size = typedInput->size();

            // Loop through image and apply max pooling process
            for (size_t i = 0; i < size; i++) {
                m_pooled_mat = max_pool_process((*typedInput)[i], m_size, m_stride, m_depth);
                //typedOutput->push_back(m_pooled_mat);
                (*typedOutput)[i] = m_pooled_mat;
            }
            
            output = static_cast<void*>(typedOutput);
        }
    };

    // Derived Layer Class that creates Flattening layer
    template <typename InputType, typename OutputType, typename DataType>
    class Flatten : public Layer<InputType,OutputType,DataType>
    {
    private:

        std::vector<DataType> m_image_vector; // Member variable to store current image vector

        //Method that flattens 3d matrix into vector
        std::vector<DataType> flatten_process(const Mat3d<DataType>& mat) override
        {
            if (mat.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return std::vector<DataType>();
            }

            int depth = mat[0][0].size();
            std::vector<DataType> output_vector;

            for (size_t d = 0; d < depth; d++) {
                for (size_t i = 0; i < mat.size(); i++) {
                    for (size_t j = 0; j < mat[i].size(); j++) {
                        output_vector.push_back(mat[i][j][d]);
                    }
                }
            }
            
            return output_vector;
        }
    
    public:
        Flatten() : Layer<InputType, OutputType, DataType>(false) {}

        // Override of layer forward method declared in base class
        void forward(void* input, void* output, std::any& weights, std::any& bias) override
        {
            if (input == nullptr) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            }

            InputType* typedInput = static_cast<InputType*>(input);
            OutputType* typedOutput = static_cast<OutputType*>(output);

            for (size_t i = 0; i < typedInput->size(); i++)
            {
                m_image_vector = flatten_process((*typedInput)[i]);
                (*typedOutput)[i] = m_image_vector;
            }
        }
    };

    // Derived Layer Class that creates Max Pooling layer
    template <typename InputType, typename OutputType, typename DataType>
    class Dense : public Layer<InputType,OutputType,DataType>
    {
    private:
        const int m_output_size; // Member Variable to hold layer output size
        const Activation m_activation_function; // Member Variable to hold passed activation function method
        std::vector<DataType> m_transformed_vector; // Member variable to store current result of linear transformation
        std::vector<DataType> m_activated_vector; // Member variable to store current result of activation function
        std::vector<DataType> m_bias_vector; // Testing only while backward propagation isn't ready
        Mat2d<DataType> m_weight_matrix; // Testing only while backward propagation isn't ready

        // Overload of activation process method to work with vectors
        std::vector<DataType> activation_process(const std::vector<DataType>& vec, const Activation& activation_function) override
        {
            if (vec.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return std::vector<DataType>();
            }

            switch (activation_function)
            {
            case RELU:
                return NumPP::relu(vec);
            case TANH:
                return NumPP::tanh(vec);
            case SIGMOID:
                return NumPP::sigmoid(vec);
            case SOFTMAX:
                return NumPP::softmax(vec);
            default:
                return std::vector<DataType>();
            }
        }

    public:
        // Class constructor to receive correct hyperparameters
        Dense(const int& output_size, const Activation& activation_function):
            Layer<InputType, OutputType, DataType>(true),
            m_output_size(output_size),
            m_activation_function(activation_function)
        {}

        // Override of layer forward method declared in base class
        void forward(void* input, void* output, std::any& weights, std::any& bias) override
        {
            if (input == nullptr) {
                std::cerr << "Error: Input is empty" << std::endl;
                return;
            }

            InputType* typedInput = static_cast<InputType*>(input);
            OutputType* typedOutput = static_cast<OutputType*>(output);

            if (!weights.has_value()) {
                weights = Mat2d<DataType>();
            }

            if (!bias.has_value()) {
                bias = std::vector<DataType>();
            }
 
            Mat2d<DataType>* weights_input = std::any_cast<Mat2d<DataType>>(&weights);
            std::vector<DataType>* biases_input = std::any_cast<std::vector<DataType>>(&bias); 

            if (weights_input->size() == 0) {
                *weights_input = NumPP::rand<DataType>((*typedInput)[0].size(), m_output_size, 0.0, 0.5);
            } 

            if (biases_input->size()  == 0) {
                *biases_input = NumPP::zeros<DataType>(m_output_size);
            }     

            for (size_t i = 0; i < typedInput->size(); i++) {
                m_transformed_vector = NumPP::add(NumPP::dot((*typedInput)[i], *weights_input), *biases_input);
                m_activated_vector = activation_process(m_transformed_vector, m_activation_function);
                (*typedOutput)[i] = m_activated_vector;
            }
        }
    };

    // Class to design neural network architecture
    class NeuralNetwork
    {
    private:
        /* Weights and biases for 2 hidden layer format in first attempt
        Mat2d<double> m_a1; // Activation output for first hidden layer
        Mat2d<double> m_a2; // Activation output for second hidden layer
        Mat2d<double> m_a3; // Activation output for output layer
        Mat2d<double> m_w1; // Weight matrix for the first hidden layer
        Mat2d<double> m_w2; // Weight matrix for the scond hidden layer
        Mat2d<double> m_w3; // Weight matrix for the output layer
        Mat2d<double> m_b1; // Bias vector for the first hidden layer
        Mat2d<double> m_b2; // Bias vector for the second hidden layer
        Mat2d<double> m_b3; // Bias vector for the output layer
        */
        
        // Member variable that store base class to determine sequence of layers in forward and backward pass
        std::vector<LayerBase*> m_layers; 
        std::vector<std::any> m_weights;
        std::vector<std::any> m_biases;
        void* m_current_input;
        void* m_current_output;
        

        // Method that applies forward propagation with given layer sequence
        void forward_pass(void* input, void* output) 
        {
            m_current_input = input;
            m_current_output = output;

            int counter = 0; // testing only

            for (auto &layer : m_layers) {
                /// !!! Still need to figure out how to clear current pointers !!!
                layer->forward(m_current_input, m_current_output, m_weights[counter], m_biases[counter]);  
                m_current_input = m_current_output; // Update input for next layer  

                if (layer->get_w_and_b()) {
                    counter++;
                }  
            }

            //output = m_current_input;
        }

    public:
        /* First attempt for neural network, commented and not removed because there may be parts we can use

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
        */
        
        // Method that creates and add layer to sequence
        void add_layer(LayerBase* layerPtr)
        {
            //void* w = nullptr;
            //void* b = nullptr;
            std::any w;
            std::any b;

            m_layers.push_back(layerPtr);

            if (layerPtr->get_w_and_b()) {   
                m_weights.push_back(w);
                m_biases.push_back(b);
            }
        }

        // Method that applies training for neural network
        template<typename T>
        Mat2d<T> fit(void *input, const size_t &epochs) 
        {
            void* output = nullptr;
            for (size_t epoch = 0; epoch < epochs; epoch++) {
                output = new void*();
                forward_pass(input, output);

                Mat2d<T>* epoch_result = static_cast<Mat2d<T>*>(output);
                auto shape = NumPP::get_shape(*epoch_result);
            
                std::cout << shape.first << std::endl;
                std::cout << shape.second << std::endl;
                DataAnalysis::display_all(*epoch_result);
            }  

            /* Debbuging

            std::cout << "==========================" << std::endl;
            std::cout << m_weights.size() << std::endl;
            std::cout << m_biases.size() << std::endl;
            std::cout << "==========================" << std::endl;
            std::cout << m_weights[0].has_value() << std::endl;
            std::cout << m_biases[0].has_value() << std::endl;
            std::cout << "==========================" << std::endl;
            std::cout << m_weights[1].has_value() << std::endl;
            std::cout << m_biases[1].has_value() << std::endl;
            std::cout << "==========================" << std::endl;
            std::cout << m_weights[2].has_value() << std::endl;
            std::cout << m_biases[2].has_value() << std::endl;

            Mat3d<_Float32>* weights_output = std::any_cast<Mat3d<_Float32>>(&m_weights[0]);       
            std::cout << weights_output->size() << std::endl; 

            for (size_t i = 0; i < weights_output->size(); i++)
            {
                DataAnalysis::display_all((*weights_output)[i]);         
            }
             
            std::vector<_Float32>* biases_outputs = std::any_cast<std::vector<_Float32>>(& m_biases[0]);
            std::cout << biases_outputs->size() << std::endl; 

            for (size_t i = 0; i < biases_outputs->size(); i++)
            {
                std::cout << (*biases_outputs)[i] << std::endl;
            } 
            
            Mat2d<_Float32>* weights_output = std::any_cast<Mat2d<_Float32>>(&m_weights[2]);
            std::cout << weights_output->size() << std::endl;
            DataAnalysis::display_all(*weights_output);

            std::vector<_Float32>* biases_outputs = std::any_cast<std::vector<_Float32>>(&m_biases[2]);
            std::cout << biases_outputs->size() << std::endl; 

            for (size_t i = 0; i < biases_outputs->size(); i++)
            {
                std::cout << (*biases_outputs)[i] << std::endl;
            }  
            */


            std::cout << "==========================" << std::endl;
            Mat2d<T>* result = static_cast<Mat2d<T>*>(output);
            auto shape = NumPP::get_shape(*result);
            
            std::cout << shape.first << std::endl;
            std::cout << shape.second << std::endl;

            return *result;
        }

        ~NeuralNetwork()
        {
            for (auto &layer : m_layers) {
                delete layer;
            }
        }

        // Further methods to be implemented
    };

#pragma endregion

#pragma region OpenCvIntegration
    // Class that transforms open cv data structures into mine for later analysis
    // This is temporary, I do not plan to use open cv in public release, this is for learning and testing purposes
    class OpenCvIntegration
    {
    private:
        // Method that caculate parameters necessary to apply warp perspective method
        static cv::Mat calc_params_for_warp_perspective(const cv::Mat& src_image, const int& adjustable_center_x = 0, 
                                                        const int& reduction_factor = 3, const bool& rectangular_perspective = false)
        {
            if (src_image.empty()) {
                std::cerr << "Error: Input is null" << std::endl;
                return cv::Mat();
            }

            int xfd = (550 / reduction_factor); // Horizontal displacement in relation to image center
            int yf = (450 / reduction_factor); // Veritcal position of origin points
            int offset_x = 0; // Horizontal displacement of origin points in relation to image borders
            int img_height = src_image.rows;
            int img_width = src_image.cols;

            // Get horizontal position of image center adding a adjustable displacement
            int center_x = (img_width / 2) + adjustable_center_x;

            std::vector<cv::Point2f> src(4); // Create vector of Point2f type to store origin coordinates
            std::vector<cv::Point2f> dst(4); // Create vector of Point2f type to store destination coordinates
            
            if (rectangular_perspective) {
                src[0] = cv::Point2f(offset_x, img_height); // Top left
                src[1] = cv::Point2f(offset_x, yf); // Bottom left
                src[2] = cv::Point2f((img_width - offset_x), yf); // Bottom right
                src[3] = cv::Point2f((img_width - offset_x), img_height); // Top right

                dst[0] = cv::Point2f(offset_x, img_height);
                dst[1] = cv::Point2f(offset_x, yf); 
                dst[2] = cv::Point2f((img_height - offset_x), yf); 
                dst[3] = cv::Point2f((img_height - offset_x), img_height);

                return cv::getPerspectiveTransform(src, dst);
            }

            src[0] = cv::Point2f(offset_x, img_height); // Top left
            src[1] = cv::Point2f((center_x - xfd), yf); // Bottom left
            src[2] = cv::Point2f((center_x + xfd), yf); // Bottom right
            src[3] =cv::Point2f((img_width - offset_x), img_height); // Top right

            dst[0] = cv::Point2f(offset_x, img_height);
            dst[1] = cv::Point2f(offset_x, yf);
            dst[2] = cv::Point2f((img_height - offset_x), yf);
            dst[3] = cv::Point2f((img_height - offset_x), img_height);

            return cv::getPerspectiveTransform(src, dst);
        }

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
    
        // Method that pre-process image based on parameters
        static cv::Mat pre_process_image(const cv::Mat* src_image_ptr, const bool& resize, const std::pair<size_t, size_t>& shape, const bool& gray_scale, 
                                            const bool& apply_blurring, const bool& apply_edge_detection)
        {
            if (src_image_ptr->empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return cv::Mat();
            }
            
            cv::Mat p_image = *src_image_ptr; // Create variable to store processed image data

            if (resize) {
                cv::resize(p_image, p_image, cv::Size(shape.first, shape.second)); // Re-scale image based on shape parameter
            }

            if (gray_scale) {
                cv::cvtColor(p_image, p_image, cv::COLOR_BGR2GRAY); // Convert image to gray scale
            }

            if (apply_blurring) {
                cv::GaussianBlur(p_image, p_image, cv::Size(5, 5), 0);
            }

            if (apply_edge_detection) {
                cv::Canny(p_image, p_image, 50, 250);
            }

            return p_image;
        }
    public:
        // Struct to hold parameters for prepare_training_data method
        struct TrainingDataParameters
        {
            const std::string dir_path;
            const bool resize = false;
            const std::pair<size_t, size_t> shape = {240, 240};
            const bool gray_scale = false;
            const bool apply_blurring = false;
            const bool apply_edge_detection = false;

            TrainingDataParameters(const std::string& dir_path, const bool& resize, const std::pair<size_t, size_t>& shape, 
            const bool& gray_scale, const bool& apply_blurring, const bool& apply_edge_detection):
                dir_path(dir_path),
                resize(resize),
                shape(shape),
                gray_scale(gray_scale),
                apply_blurring(apply_blurring), 
                apply_edge_detection(apply_edge_detection)
            
            {}
        };

        // Method that applies Warp perspective to image and return image in 
        // Bird's Eye View (bev) perspective
        static cv::Mat change_perspective_to_bev(const cv::Mat& src_image)
        {
            if (src_image.empty()) {
                std::cerr << "Error: Input is null" << std::endl;
                return cv::Mat();
            }

            cv::Mat bev_image;
            cv::Mat parameters = calc_params_for_warp_perspective(src_image, 0, 3, false);
            cv::Size img_dimensions(src_image.cols, src_image.rows);
            cv::warpPerspective(src_image, bev_image, parameters, img_dimensions, cv::INTER_LINEAR);

            return bev_image;
        }

        // Get open cv color image matrix and convert it to our 3d matrix data structure
        // Default format is Blue, Green, Red
        template <typename T>
        static Mat3d<T>* convert_color_image(const cv::Mat* image) 
        {
            if (!image->data) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Image data empty" << std::endl;
                return nullptr; // Return nullptr if image data is empty
            }

            // Create a new Mat3d pointer and allocate memory for it
            Mat3d<T>* nMatPtr = new Mat3d<T>(image->rows, Mat2d<T>(image->cols, std::vector<T>(3)));
            // Create pointer to hold open cv pixel values
            cv::Vec3b* cvPixelPtr = new cv::Vec3b(3);
            // Iterate over each pixel in the given image
            for (int i = 0; i < image->rows; i++) {
                for (int j = 0; j < image->cols; j++) {
                    // Assing current pixel data to pointer
                    (*cvPixelPtr) = image->at<cv::Vec3b>(i, j);
                    // Access the corresponding pixel in the new matrix and copy pixel values
                    (*nMatPtr)[i][j] = {static_cast<T>((*cvPixelPtr)[0]),
                                        static_cast<T>((*cvPixelPtr)[1]), 
                                        static_cast<T>((*cvPixelPtr)[2])};
                }
            }

            delete cvPixelPtr;
            // Return the pointer to the final matrix
            return nMatPtr;
        }

        // Overload of convert_image method for gray images
        template <typename T>
        static Mat3d<T>* convert_gray_image(const cv::Mat* image) 
        {
            if (!image->data) {
                // Display error that informs data matrix is empty
                std::cerr << "Error: Image data empty" << std::endl;
                return nullptr; // Return nullptr if image data is empty
            }

            // Create a new Mat3d pointer and allocate memory for it
            Mat3d<T>* nMatPtr = new Mat3d<T>(image->rows, Mat2d<T>(image->cols, std::vector<T>(1)));
            // Create pointer to hold open cv pixel values
            uchar* cvPixelPtr = new uchar(1);
            // Iterate over each pixel in the given image
            for (int i = 0; i < image->rows; i++) {
                for (int j = 0; j < image->cols; j++) {
                    // Assing current pixel data to pointer
                    (*cvPixelPtr) = image->at<uchar>(i, j);
                    // Access the corresponding pixel in the new matrix and copy pixel values
                    (*nMatPtr)[i][j][0] = static_cast<T>((*cvPixelPtr));
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
        
        // Method to convert 3d matrix into open cv color matrix
        template <typename T>
        static cv::Mat get_open_cv_color_mat(const Mat3d<T>* matPtr)
        {
            if (matPtr == nullptr) {
                std::cerr << "Error: Input is empty" << std::endl;
                return cv::Mat();
            }

            // Get dimensions of the input matrix
            int rows = static_cast<int>(matPtr->size());
            int cols = static_cast<int>((*matPtr)[0].size());
            int depth = static_cast<int>((*matPtr)[0][0].size());

            // Determine OpenCV data type based on the template parameter T
            int cvDataType;
            if (std::is_same<T, uchar>::value) {
                cvDataType = CV_8UC(depth);
                cv::Mat opencv_mat(rows, cols, cvDataType);

                for (int d = 0; d < depth; d++) {
                    for (int i = 0; i < rows; i++) {
                        for (int j = 0; j < cols; j++) {
                            if (depth == 1) {
                                opencv_mat.at<cv::Vec<T, 1>>(i, j)[d] = static_cast<T>((*matPtr)[i][j][d]);
                            } else if (depth == 2) {
                                opencv_mat.at<cv::Vec<T, 2>>(i, j)[d] = static_cast<T>((*matPtr)[i][j][d]);
                            } else {
                                opencv_mat.at<cv::Vec<T, 3>>(i, j)[d] = static_cast<T>((*matPtr)[i][j][d]);
                            }
                        }
                    }
                }

                return opencv_mat;
            } 
            else {
                cvDataType = CV_32FC(depth);

                cv::Mat opencv_mat(rows, cols, cvDataType);

                for (int d = 0; d < depth; d++) {
                    for (int i = 0; i < rows; i++) {
                        for (int j = 0; j < cols; j++) {
                            if (depth == 1) {
                                opencv_mat.at<cv::Vec<float, 1>>(i, j)[d] = static_cast<float>((*matPtr)[i][j][d]);
                            } else if (depth == 2) {
                                opencv_mat.at<cv::Vec<float, 2>>(i, j)[d] = static_cast<float>((*matPtr)[i][j][d]);
                            } else {
                                opencv_mat.at<cv::Vec<float, 3>>(i, j)[d] = static_cast<float>((*matPtr)[i][j][d]);
                            }
                        }
                    }
                }

                return opencv_mat;
            } 
        }

        // Method to convert 3d matrix into open cv matrix
        // Added conv_channel parameter to test if all convolutional feature maps are being
        // generated correctly
        template <typename T>
        static cv::Mat get_open_cv_gray_mat(const Mat3d<T>& matPtr, int conv_channel = 0)
        {
            if (matPtr.empty()) {
                std::cerr << "Error: Input is empty" << std::endl;
                return cv::Mat();
            }

            // Get dimensions of the input matrix
            int rows = matPtr.size();
            int cols = matPtr[0].size();

            cv::Mat opencv_mat(rows, cols, CV_8UC1);

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    opencv_mat.at<uchar>(i, j) = static_cast<uchar>(matPtr[i][j][conv_channel]);
                }
            }

            return opencv_mat;
        }
        
        // Method that iterates through all images in given directory and return Mat4d with all images converted to Mat3d
        template <typename T>
        static Mat4d<T> prepare_training_data(const TrainingDataParameters& params)
        {
            if (!std::filesystem::exists(params.dir_path) && !std::filesystem::is_directory(params.dir_path)) {
                std::cerr << "Error: Directory not valid" << std::endl;
                return Mat4d<T>();
            }

            cv::Mat* imagePtr = new cv::Mat(); // Create pointer to store open cv matrices read from each image    
            Mat4d<T> training_data_mat; // Create 4d matrix that will hold all images matrices

            if (!(params.resize || params.gray_scale || params.apply_blurring || params.apply_edge_detection)) {
                // Iterate over the contents of the directory
                for (const auto &entry : std::filesystem::directory_iterator(params.dir_path)) {
                    std::cout << entry.path() << std::endl;
                    *imagePtr = cv::imread(entry.path()); // Read current entry image in directory into pointer
                    Mat3d<T>* converted_image_mat = convert_color_image<T>(imagePtr); // Convert opencv matrix to Mat3d
                    training_data_mat.push_back(*converted_image_mat); // Add Mat3d into training data

                    delete converted_image_mat;
                }

                delete imagePtr;

                return training_data_mat;
            }

            cv::Mat p_image; // Create pointer to store image after pre-processing

            // Iterate over the contents of the directory
            for (const auto &entry : std::filesystem::directory_iterator(params.dir_path)) {
                std::cout << entry.path() << std::endl;
                *imagePtr = cv::imread(entry.path()); // Read current entry image in directory into pointer
                p_image = pre_process_image(imagePtr, params.resize, params.shape, params.gray_scale, params.apply_blurring, params.apply_edge_detection);
                Mat3d<T>* converted_image_mat = convert_gray_image<T>(&p_image); // Convert opencv matrix to Mat3d
                training_data_mat.push_back(*converted_image_mat); // Add Mat3d into training data

                delete converted_image_mat;
            }

            delete imagePtr;

            return training_data_mat;
        
        }

        // Further methods to be implemented
    };

#pragma endregion

    // Further classes to be implemented
}