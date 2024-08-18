#include "../src/mlpp.hpp"
#include "catch.hpp"
#include "testUtils.hpp"

using namespace MLPP; 
using namespace Utils;
using namespace Benchmark;

TEST_CASE("NEURAL_NETWORK_TRAINING")
{
    SECTION("ORIGINAL_ARCHITECTURE")
    {
        auto params = OpenCvIntegration::TrainingDataParameters("../tests/Training", true, {28, 28}, true, true, true);

        auto training_data = OpenCvIntegration::prepare_training_data<double>(params); 

        std::cout << "Input shape = ";
        std::cout << "(" << std::to_string(training_data.size()) << "," << std::to_string(training_data[0].size()) << ",";
        std::cout << std::to_string(training_data[0][0].size()) << "," << std::to_string(training_data[0][0][0].size()) << ")" << std::endl;

        // Data normalization, since max value is 255, all values will be between 0 and 1
        for (size_t i = 0; i < training_data.size(); i++) {
            for (size_t j = 0; j < training_data[i].size(); j++) {
                for (size_t k = 0; k < training_data[i][j].size(); k++) {
                    for (size_t l = 0; l < training_data[i][j][k].size(); l++) {
                        training_data[i][j][k][l] /= 255.0;
                    }
                }
            }
        } 

        NeuralNetwork model;
        // testing                                   0       1       2       3       4       5       6       7       8
        const Mat2d<double> hot_encoded_labels = {{0,0,1},{0,1,0},{1,0,0},{0,1,0},{0,1,0},{0,0,1},{0,1,0},{0,1,0},{1,0,0}};

        model.add_layer(new Conv2D<Mat4d<double>, Mat4d<double>, double>(5, 3, RELU, SAME));
        model.add_layer(new AveragePooling2d<Mat4d<double>, Mat4d<double>, double>(2,2));
        model.add_layer(new Flatten<Mat4d<double>, Mat2d<double>, double>());
        model.add_layer(new Dense<Mat2d<double>, Mat2d<double>, double>(120, RELU));
        model.add_layer(new Dense<Mat2d<double>, Mat2d<double>, double>(3, SOFTMAX));

        auto t1 = startBenchmark();
        Mat2d<double> result = model.fit<double>(training_data, hot_encoded_labels, 9, 150, 0.001);
        auto t2 = stopBenchmark();

        std::cout << "================================" << std::endl;
        std::cout << getDuration(t1, t2, Seconds) << std::endl;
        std::cout << "================================" << std::endl;

        auto shape = NumPP::get_shape(result);

        std::cout << "Output layer shape = ";
        std::cout << "(" << shape.first << ", " << shape.second << ")" << std::endl; 

        for (size_t i = 0; i < result.size(); i++) {
            std::cout << i << ", ";
            for (size_t j = 0; j < result[i].size(); j++) {
                std::cout << std::to_string(result[i][j]) << ", ";
            }
            std::cout << std::endl;
        } 

        for (size_t i = 0; i < shape.first; i++)
        {
            // Extract detected class from result and compare to original class
            auto og_class = NumPP::get_max_element_pos(hot_encoded_labels[i]);
            auto pred_class = NumPP::get_max_element_pos(result[i]);

            REQUIRE(og_class == pred_class);
        }
        
    }

   /* SECTION("AC_CAR_ARCHITECTURE")
    {
        auto params = OpenCvIntegration::TrainingDataParameters("/home/muzaodamassa/Downloads/AC_Training_Data", true, {28, 28}, true, true, true);

        auto training_data = OpenCvIntegration::prepare_training_data<double>(params); 

        std::cout << "Input shape = ";
        std::cout << "(" << std::to_string(training_data.size()) << "," << std::to_string(training_data[0].size()) << ",";
        std::cout << std::to_string(training_data[0][0].size()) << "," << std::to_string(training_data[0][0][0].size()) << ")" << std::endl;

        // Data normalization, since max value is 255, all values will be between 0 and 1
        for (size_t i = 0; i < training_data.size(); i++) {
            for (size_t j = 0; j < training_data[i].size(); j++) {
                for (size_t k = 0; k < training_data[i][j].size(); k++) {
                    for (size_t l = 0; l < training_data[i][j][k].size(); l++) {
                        training_data[i][j][k][l] /= 255.0;
                    }
                }
            }
        } 

        NeuralNetwork model;
        auto true_labels = DataAnalysis::one_hot_label_encoding<double>("/home/muzaodamassa/Downloads/AC_Training_Data");

        DataAnalysis::display_shape(true_labels);

        model.add_layer(new Conv2D<Mat4d<double>, Mat4d<double>, double>(5, 3, RELU, SAME));
        model.add_layer(new AveragePooling2d<Mat4d<double>, Mat4d<double>, double>(2,2));
        model.add_layer(new Flatten<Mat4d<double>, Mat2d<double>, double>());
        model.add_layer(new Dense<Mat2d<double>, Mat2d<double>, double>(120, RELU));
        model.add_layer(new Dense<Mat2d<double>, Mat2d<double>, double>(3, SOFTMAX));

        auto t1 = startBenchmark();
        Mat2d<double> result = model.fit<double>(training_data, true_labels, 150, 0.001);
        auto t2 = stopBenchmark();

        std::cout << "================================" << std::endl;
        std::cout << getDuration(t1, t2, Seconds) << std::endl;
        std::cout << "================================" << std::endl;

        auto shape = NumPP::get_shape(result);

        std::cout << "Output layer shape = ";
        std::cout << "(" << shape.first << ", " << shape.second << ")" << std::endl; 

        for (size_t i = 0; i < result.size(); i++) {
            std::cout << i << ", ";
            for (size_t j = 0; j < result[i].size(); j++) {
                std::cout << std::to_string(result[i][j]) << ", ";
            }
            std::cout << std::endl;
        } 

        for (size_t i = 0; i < shape.first; i++)
        {
            // Extract detected class from result and compare to original class
            auto og_class = NumPP::get_max_element_pos(true_labels[i]);
            auto pred_class = NumPP::get_max_element_pos(result[i]);

            REQUIRE(og_class == pred_class);
        }
        
    } */

}