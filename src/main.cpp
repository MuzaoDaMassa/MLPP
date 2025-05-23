#include <iostream>
#include "mlpp.hpp"
#include "../tests/testUtils.hpp"

using namespace std;
using namespace MLPP;
using namespace Benchmark;

// Mini tests for data analysis unit
int main1() 
{ 
    auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
    auto data2 = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
    auto cData = DataAnalysis::matrix_converter<float>(data);
    // Print data
    cout << "========================================" << endl;
    //DataAnalysis::display_all(data);
    cout << "========================================" << endl;
    vector<int> rows{0, 10, 20, 30, 40, 50, 60, 70, 80, 90};
    //DataAnalysis::display_rows(data, rows);
    cout << "========================================" << endl;
    vector<int> cols{2, 4, 6, 8};
    //DataAnalysis::display_columns(data, cols);
    //vector<int> pos1 {100, 8};
    //auto el = find_by_pos(data, pos1);

    vector<string> testR{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    vector<string> testR0{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    vector<string> testR1{"3", "4", "5", "3", "4", "5", "6", "7", "8", "9"};
    vector<string> testR2{"6", "7", "8", "3", "4", "5", "6", "7", "8", "9"};
    vector<string> testR3{"9", "10", "11", "3", "4", "5", "6", "7", "8", "9"};
    vector<string> testC(data.size(), "0");
    Mat2d<string> testAddR {{testR}, {testR1}, {testR2}, {testR3}};
    Mat2d<string> testAddC {{testR}, {testR1}, {testC}};
    Mat2d<int> testAddCandR {{rows}, {cols}};
    //DataAnalysis::matrix_formatter<string>(data, ROWANDCOLUMN, testAddCandR ,testAddR);
    // auto pos = find<string>(data, "http://www.hatfield-saunders.net/");
    // auto pos = find<string>(data, "2020-03-11");

    auto hot_encoded_label = DataAnalysis::gen_hot_encoded_label<double>(3, 2);

    DataAnalysis::display_all(hot_encoded_label);

    cout << "========================================" << endl;
    //DataAnalysis::display_rows(data, rows);
    //DataAnalysis::display_columns(data, cols);
    // cout << el << endl;
    // cout << pos[0] << ", " << pos[1] << endl;
    //DataAnalysis::display_all(data);
    //DataAnalysis::display_head(data, 20);
    //DataAnalysis::display_bottom(data, 5);

    auto check = DataAnalysis::verify_integrity_between_matrices(data, data2);
    std::cout << check << std::endl;

    return 0;
}

// Mini tests for NumPP unit
int main2()
{
    auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/hw_100.csv");
    auto cData = DataAnalysis::matrix_converter<double>(data);    

    //auto r = NumPP::rand<float>(15, 15, 20.0, 5.0);
    //auto r = NumPP::gen_square_matrix(4, 10);
    //auto test = NumPP::get_center(r);
    //Mat2d<double> T = NumPP::transpose(cData);
    //Mat2d<double> r1 = NumPP::dot(T, cData);

    //Mat2d<int16_t> a {{-1,-1,-1}, {0,1,-1}, {0,1,1}};
    //Mat2d<int16_t> b {{0,0,0}, {0,156,155}, {0,153,154}};
    //Mat2d<int16_t>* aPtr = new Mat2d<int16_t>(a);
    //Mat2d<int16_t>* bPtr = new Mat2d<int16_t>(b);
    //auto r = NumPP::sum_mat_mul_matching_elements(aPtr, bPtr);
    //Mat2d<double> r = NumPP::scalar_mat_mul<int16_t, double>(b, 2.0);
    //cout << to_string(r) << endl;

    //vector<u_int8_t> v {255, 255, 255};
    //cout << to_string(NumPP::get_sum_of_vector<u_int8_t, u_int16_t>(v)) << endl;

    //DataAnalysis::display_all(r);
    //std::cout << to_string(NumPP::get_average(r)) << std::endl;

    Mat2d<int16_t> a {{1,2,3},{4,5,6},{7,8,9}};
    auto v_flip = NumPP::flip_matrix(a, true);
    auto h_flip = NumPP::flip_matrix(a, false);
    auto h_and_v_flip = NumPP::flip_matrix(a, true, true);
    DataAnalysis::display_all(a);
    std::cout << "-------------------------------------------------------------" << std::endl;
    DataAnalysis::display_all(v_flip);
    std::cout << "-------------------------------------------------------------" << std::endl;
    DataAnalysis::display_all(h_flip);
    std::cout << "-------------------------------------------------------------" << std::endl;
    DataAnalysis::display_all(h_and_v_flip);
    
    
    
    //DataAnalysis::display_all(r1);

    return 0;
}

// Open CV Image testing
int main3()
{
    /* cv::Mat* imagePtr = new cv::Mat(cv::imread("../tests/Images/Pista1.jpg"));
	if (!imagePtr->data) { 
		std::cerr << "Error: No image data" << std::endl; 
		return -1; 
	} 

    //cv::Mat resizedImage;  
    //cv::resize(*imagePtr, resizedImage, cv::Size(420, 240));
    //cv::Mat* rImagePtr = new cv::Mat(resizedImage);

    //cv::Mat grayImage;
    //cv::cvtColor(*rImagePtr, grayImage, cv::COLOR_BGR2GRAY);
    //cv::Mat* grayPtr = new cv::Mat(grayImage);

    auto start = startBenchmark();
    auto nImagePtr = OpencvIntegration::convert_color_image<float>(imagePtr);
    auto end = stopBenchmark();

    cout << getDuration(start, end, Seconds) << endl;

    // Access the element at row 'i' and column 'j'
    std::vector<float>& pixel = (*nImagePtr)[150][150];
    cv::Vec3b& cvPixel = imagePtr->at<cv::Vec3b>(150, 150);
    // Now you can access individual channels of the pixel
    uint8_t blue = pixel[0];
    uint8_t green = pixel[1];
    uint8_t red = pixel[2];
    
    cout << cvPixel << endl;
    cout << "[" << to_string(pixel[0]) << ", " << to_string(pixel[1]) << ", " << to_string(pixel[2]) << "]" << endl;
    cout << endl;

    delete imagePtr;
    //delete rImagePtr; */

    cv::Mat* imagePtr = new cv::Mat(cv::imread("/home/muzaodamassa/Documents/AC_Training_Data/High_Angle_Data_3/Class_1(Straight)/frame1000.jpg"));
	if (!imagePtr->data) { 
		std::cerr << "Error: No image data" << std::endl; 
		return -1; 
	} 

    cv::namedWindow("Original Image", cv::WINDOW_FREERATIO);
    cv::imshow("Original Image", *imagePtr);
    cv::waitKey(0);   
    
    // Read the image
    cv::Mat img = cv::imread("/home/muzaodamassa/Documents/AC_Training_Data/High_Angle_Data_3/Class_1(Straight)/frame1000.jpg");
    if (img.empty()) {
        std::cerr << "Error: Unable to open image!" << std::endl;
        return -1;
    }

    // Define the ROI coordinates
    int x_start = 0;  // x-coordinate of the top-left corner
    int y_start = 280;   // y-coordinate of the top-left corner
    int x_end = 1920;    // x-coordinate of the bottom-right corner
    int y_end = 1080;    // y-coordinate of the bottom-right corner

    // Define the rectangle for the region of interest
    cv::Rect roi(x_start, y_start, x_end - x_start, y_end - y_start);

    // Crop the image
    cv::Mat croppedImage = img(roi);

    // Display the result
    cv::imshow("Cropped Image", croppedImage);
    cv::waitKey(0); // Wait for a key press before closing the image window

    return 0;

    //cv::Mat resizedImage;  
    //cv::resize(*imagePtr, resizedImage, cv::Size(1000, 1000));

    //cv::cvtColor(*imagePtr, *imagePtr, cv::COLOR_BGR2GRAY);
    //cv::GaussianBlur(*imagePtr, *imagePtr, cv::Size(5, 5), 0);
    //cv::Canny(*imagePtr, *imagePtr, 300, 500);
    //cv::resize(*imagePtr, *imagePtr, cv::Size(512,512)); 
    //cv::namedWindow("Processed Image", cv::WINDOW_KEEPRATIO);
    //cv::imshow("Processed Image", *imagePtr);
    //cv::waitKey(0);

    //cv::Mat resizedImage;  
    //cv::resize(*imagePtr, resizedImage, cv::Size(240, 240));

    //cv::Mat grayImage;
    //cv::cvtColor(resizedImage, grayImage, cv::COLOR_BGR2GRAY);

    //Mat3d<double>* mat_0 = OpenCvIntegration::convert_gray_image<double>(&grayImage);

    //cv::Mat restored_image = OpenCvIntegration::get_open_cv_gray_mat<double>(*mat_0);

    //cv::namedWindow("Gray Scale Image", cv::WINDOW_AUTOSIZE);
    //cv::imshow("Gray Scale Image", restored_image);
    //cv::waitKey(0);

    //cv::Mat bev_image = OpenCvIntegration::change_perspective_to_birds_eye_view(*imagePtr);

    //cv::namedWindow("Birds Eye View", cv::WINDOW_KEEPRATIO);
    //cv::imshow("Birds Eye View", bev_image);
    //cv::waitKey(0);
}

// Open CV Video capture testing
int main4()
{
    // Open default camera
    cv::VideoCapture cap(0);

    // Check if camera opened successfully
    if (!cap.isOpened())
    {
        cerr << "Error opening video stream or file" << endl;
        return -1;
    }

    // Create a window to display the video
    cv::namedWindow("Video Stream", cv::WINDOW_AUTOSIZE);
    
	// Set capture resolution
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 64);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 64);

	// Variables for fps calculation
    double fps;
    cv::TickMeter tm;

    while (true)
    {
		tm.start();

		// Capture frame-by-frame
		cv::Mat frame;
        cap >> frame;

        cv::Mat* framePtr = new cv::Mat(frame);
		
		// If the frame is empty, break immediately
		if (frame.empty())
		{
			break;
		}

/*         auto matPtr = OpenCvIntegration::convert_color_image<float>(framePtr);

        // Access the element at row 'i' and column 'j'
        std::vector<float> &pixel = (*matPtr)[150][150];

        // Now you can access individual channels of the pixel
        float blue = pixel[0];
        float green = pixel[1];
        float red = pixel[2];
        cout << framePtr->at<cv::Vec3b>(150, 150) << endl;
        cout << "[" << to_string(pixel[0]) << ", " << to_string(pixel[1]) << ", " << to_string(pixel[2]) << "]" << endl;
 */
		// Convert the frame to grayscale
		//cv::Mat gray;
		//cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Calculate FPS
		tm.stop();
		fps = 1.0 / tm.getTimeSec();
		tm.reset();

		// Put FPS text on the frame
		cv::putText(frame, "FPS: " + to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

		// Dispay the resulting frame
		cv::imshow("Video Stream", frame);

		// Press q to exit the loop
		if (cv::waitKey(1) == 'q')
		{			
			break;
		}	
    }

	// Whene everything is done, release the video capture and video writer objects
	cap.release();

    // Close all windows
	cv::destroyAllWindows();

    return 0;	
}

// Mini tests for Sequential neural network with open cv (AC_Project)
int main()
{
    auto params = OpenCvIntegration::TrainingDataParameters("/home/muzaodamassa/Documents/AC_Training_Data/Trimmed_Low_Angle_Data_3", false, true, {24,24}, true, false, false);

    Mat4d<double> training_data = OpenCvIntegration::prepare_training_data<double>(params); 

    cout << "Input shape = ";
    cout << "(" << to_string(training_data.size()) << "," << to_string(training_data[0].size()) << ",";
    cout << to_string(training_data[0][0].size()) << "," << to_string(training_data[0][0][0].size()) << ")" << endl;

    // Data normalization, since max value is 255, all values will be between 0 and 1
    for (size_t i = 0; i < training_data.size(); i++) {
        for (size_t j = 0; j < training_data[i].size(); j++) {
            for (size_t k = 0; k < training_data[i][j].size(); k++) {
                for (size_t l = 0; l < training_data[i][j][k].size(); l++) {
                    training_data[i][j][k][l] /= 255.0;
                    if (training_data[i][j][k][l] < 0 || training_data[i][j][k][l] > 1.0){
                        throw std::runtime_error("Error: Data Normalization Failed");
                    }
                }
            }
        }
    } 

    NeuralNetwork model;

    auto true_labels = DataAnalysis::one_hot_label_encoding<double>("/home/muzaodamassa/Documents/AC_Training_Data/Trimmed_Low_Angle_Data_3");

    DataAnalysis::display_shape(true_labels);

    model.add_layer(new Conv2D<Mat4d<double>, Mat4d<double>, double>(16, 3, RELU, SAME, true));
    model.add_layer(new AveragePooling2d<Mat4d<double>, Mat4d<double>, double>(2,2));
    //model.add_layer(new Conv2D<Mat4d<double>, Mat4d<double>, double>(16, 3, RELU, SAME));
    //model.add_layer(new AveragePooling2d<Mat4d<double>, Mat4d<double>, double>(2,2));
    model.add_layer(new Flatten<Mat4d<double>, Mat2d<double>, double>());
    model.add_layer(new Dense<Mat2d<double>, Mat2d<double>, double>(128, RELU));
    model.add_layer(new Dense<Mat2d<double>, Mat2d<double>, double>(64, RELU));
    model.add_layer(new Dense<Mat2d<double>, Mat2d<double>, double>(32, RELU));
    model.add_layer(new Dense<Mat2d<double>, Mat2d<double>, double>(3, SOFTMAX));

    auto t1 = startBenchmark();
    Mat2d<double> result = model.fit<double>(training_data, true_labels, 2, 25, 1.0E-5);
    auto t2 = stopBenchmark();

    cout << "================================" << endl;
    cout << getDuration(t1, t2, Seconds) << endl;
    cout << "================================" << endl;

    auto shape = NumPP::get_shape(result);

    cout << "Output layer shape = ";
    cout << "(" << shape.first << ", " << shape.second << ")" << endl; 

    /*for (size_t i = 0; i < result.size(); i++) {
        cout << i << ", ";
        for (size_t j = 0; j < result[i].size(); j++) {
            cout << std::to_string(result[i][j]) << ", ";
        }
        cout << endl;
    } 
     */
    
    auto acc = model.get_accuracy<double>(result, true_labels);

    std::cout << "Predictions Accuracy: " << acc*100 << "%" << std::endl;

    string fileName = "new_road_vision_low3_25.bin";
    model.export_model(fileName);

    return 0;
    
    /* Halt Execution
    char input;
    std::cout << "Program is running. Press 'q' to quit, or any other key to continue." << endl;
    
    while (true) {
        input = cin.get();

        if (input == 'q' || input == 'Q') {
            cout << "Program ended" << endl;
            return 0;
        }
        if (input != 'q' || input != 'Q') {
            break;
        }
    }
    
    cout << "Program resumed" << endl;
    return 0;
    */
}

// GS Artifical Intelligence
int main6()
{
    NeuralNetwork model;

    auto params = OpenCvIntegration::TrainingDataParameters("/home/muzaodamassa/Documents/School/Asignments/GS_1/AI/GS_Dataset/train", false, true, {14, 14}, true, false, false);

    auto training_data = OpenCvIntegration::prepare_training_data<double>(params); 

    DataAnalysis::display_shape(training_data);

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
    
    auto hot_encoded_labels = DataAnalysis::one_hot_label_encoding<double>("/home/muzaodamassa/Documents/School/Asignments/GS_1/AI/GS_Dataset/train");

    DataAnalysis::display_shape(hot_encoded_labels);
    DataAnalysis::display_head(hot_encoded_labels);

    model.add_layer(new Conv2D<Mat4d<double>, Mat4d<double>, double>(3, 3, RELU, SAME));
    model.add_layer(new MaxPooling2D<Mat4d<double>, Mat4d<double>, double>(2,2));
    model.add_layer(new Flatten<Mat4d<double>, Mat2d<double>, double>());
    model.add_layer(new Dense<Mat2d<double>, Mat2d<double>, double>(120, RELU));
    model.add_layer(new Dense<Mat2d<double>, Mat2d<double>, double>(3, SOFTMAX));

    auto t1 = startBenchmark();
    Mat2d<double> result = model.fit<double>(training_data, hot_encoded_labels, 9, 1, 0.01);
    auto t2 = stopBenchmark();

    cout << "================================" << endl;
    cout << getDuration(t1, t2, Seconds) << endl;
    cout << "================================" << endl;
    
    DataAnalysis::display_head(result);
 
    return 0;
}

// Mini tests for Sequential neural network with open cv (Multithreading)
int main7()
{
    auto params = OpenCvIntegration::TrainingDataParameters("/home/muzaodamassa/Downloads/training_set", false, true, {8,8}, true, false, false);

    auto training_data = OpenCvIntegration::prepare_training_data<double>(params); 

    cout << "Input shape = ";
    cout << "(" << to_string(training_data.size()) << "," << to_string(training_data[0].size()) << ",";
    cout << to_string(training_data[0][0].size()) << "," << to_string(training_data[0][0][0].size()) << ")" << endl;

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

    auto true_labels = DataAnalysis::one_hot_label_encoding<double>("/home/muzaodamassa/Downloads/training_set");

    DataAnalysis::display_shape(true_labels);
    
    model.add_layer(new Conv2D<Mat4d<double>, Mat4d<double>, double>(5, 3, RELU, SAME));
    model.add_layer(new AveragePooling2d<Mat4d<double>, Mat4d<double>, double>(2,2));
    model.add_layer(new Flatten<Mat4d<double>, Mat2d<double>, double>());
    model.add_layer(new Dense<Mat2d<double>, Mat2d<double>, double>(120, RELU));
    model.add_layer(new Dense<Mat2d<double>, Mat2d<double>, double>(2, SOFTMAX));

    auto t1 = startBenchmark();
    //Mat2d<double> result = model.fit_multithreaded<double>(training_data, true_labels, 64, 1, 0.00001);
    auto t2 = stopBenchmark();

    cout << "================================" << endl;
    cout << getDuration(t1, t2, Seconds) << endl;
    cout << "================================" << endl;

    //auto shape = NumPP::get_shape(result);

    cout << "Output layer shape = ";
    //cout << "(" << shape.first << ", " << shape.second << ")" << endl; 

    /*for (size_t i = 0; i < result.size(); i++) {
        cout << i << ", ";
        for (size_t j = 0; j < result[i].size(); j++) {
            cout << std::to_string(result[i][j]) << ", ";
        }
        cout << endl;
    } 
     */
    
    //auto acc = model.get_accuracy<double>(result, true_labels);

    //std::cout << "Predictions Accuracy: " << acc*100 << "%" << std::endl;

    return 0;
    
    /* Halt Execution
    char input;
    std::cout << "Program is running. Press 'q' to quit, or any other key to continue." << endl;
    
    while (true) {
        input = cin.get();

        if (input == 'q' || input == 'Q') {
            cout << "Program ended" << endl;
            return 0;
        }
        if (input != 'q' || input != 'Q') {
            break;
        }
    }
    
    cout << "Program resumed" << endl;
    return 0;
    */
}

// Mini tests for open cv integration module
int main8()
{
    // Low Angle
    std::string path_to_video = "/home/muzaodamassa/Downloads/VideosPista/Pista/Curva_exterior_direita_1 (trimmed).mp4";
    std::string save_location = "/home/muzaodamassa/Documents/AC_Training_Data/Trimmed_Low_Angle_Data_3/Class2(Right)";

    OpenCvIntegration::convert_video_to_images(path_to_video, save_location);

    return 0;
}

// Mini tests for creating neural network model (R2_D2)
int main9()
{
    auto params = OpenCvIntegration::TrainingDataParameters("/home/muzaodamassa/Downloads/archive/Test", false, false, {28,28}, true, true, false);

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
    //const Mat2d<double> hot_encoded_labels = {{0,0,1},{0,1,0},{1,0,0},{0,1,0},{0,1,0},{0,0,1},{0,1,0},{0,1,0},{1,0,0}};
    auto true_labels = DataAnalysis::one_hot_label_encoding<double>("/home/muzaodamassa/Downloads/archive/Test");
    
    DataAnalysis::display_shape(true_labels);
    
    model.add_layer(new Conv2D<Mat4d<double>, Mat4d<double>, double>(5, 3, RELU, SAME));
    model.add_layer(new AveragePooling2d<Mat4d<double>, Mat4d<double>, double>(2,2));
    model.add_layer(new Flatten<Mat4d<double>, Mat2d<double>, double>());
    model.add_layer(new Dense<Mat2d<double>, Mat2d<double>, double>(128, RELU));
    model.add_layer(new Dense<Mat2d<double>, Mat2d<double>, double>(24, SOFTMAX));
    auto t1 = startBenchmark();
    Mat2d<double> result = model.fit<double>(training_data, true_labels, 4, 100, 1.0E-4);
    auto t2 = stopBenchmark();
    std::cout << "================================" << std::endl;
    std::cout << getDuration(t1, t2, Seconds) << std::endl;
    std::cout << "================================" << std::endl;

    auto acc = model.get_accuracy<double>(result, true_labels);

    std::cout << "Predictions Accuracy: " << acc*100 << "%" << std::endl;

    // Alpha precision 40 epochs
    // Predictions Accuracy: 65.0725%

    string fileName = "model_r2_test.bin";
    model.export_model(fileName);
    return 0;
}

// Mini tests for exporting neural network model
int main10()
{
    auto params = OpenCvIntegration::TrainingDataParameters("/home/muzaodamassa/MLPP/tests/Training", false, true, {28,28}, true, false, false);

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
    //auto true_labels = DataAnalysis::one_hot_label_encoding<double>("/home/muzaodamassa/Downloads/archive/Test");
    
    //DataAnalysis::display_shape(true_labels);
    
    model.add_layer(new Conv2D<Mat4d<double>, Mat4d<double>, double>(8, 3, RELU, SAME));
    model.add_layer(new AveragePooling2d<Mat4d<double>, Mat4d<double>, double>(2,2));
    model.add_layer(new Flatten<Mat4d<double>, Mat2d<double>, double>());
    model.add_layer(new Dense<Mat2d<double>, Mat2d<double>, double>(32, RELU));
    model.add_layer(new Dense<Mat2d<double>, Mat2d<double>, double>(64, RELU));
    model.add_layer(new Dense<Mat2d<double>, Mat2d<double>, double>(3, SOFTMAX));
    auto t1 = startBenchmark();
    Mat2d<double> result = model.fit<double>(training_data, hot_encoded_labels, 9, 50, 0.001);
    auto t2 = stopBenchmark();
    std::cout << "================================" << std::endl;
    std::cout << getDuration(t1, t2, Seconds) << std::endl;
    std::cout << "================================" << std::endl;

    auto acc = model.get_accuracy<double>(result, hot_encoded_labels);

    std::cout << "Predictions Accuracy: " << acc*100 << "%" << std::endl;

    string fileName = "testing_model.bin";
    model.export_model(fileName);

    string model_path = "../build/testing_model.bin";
    model.load_model(model_path);

    return 0;
}

// Mini tests for loading and evaluating neural network models
int main11()
{
    NeuralNetwork cnn;
    cnn.load_model("../trained_models/cats_and_dogs_model_test.bin");

    auto params = OpenCvIntegration::TrainingDataParameters("/home/muzaodamassa/Downloads/test_set", false, true, {32,32}, true, false, false);

    Mat4d<double> testing_data = OpenCvIntegration::prepare_training_data<double>(params); 

    cout << "Input shape = ";
    cout << "(" << to_string(testing_data.size()) << "," << to_string(testing_data[0].size()) << ",";
    cout << to_string(testing_data[0][0].size()) << "," << to_string(testing_data[0][0][0].size()) << ")" << endl;

    // Data normalization, since max value is 255, all values will be between 0 and 1
    for (size_t i = 0; i < testing_data.size(); i++) {
        for (size_t j = 0; j < testing_data[i].size(); j++) {
            for (size_t k = 0; k < testing_data[i][j].size(); k++) {
                for (size_t l = 0; l < testing_data[i][j][k].size(); l++) {
                    testing_data[i][j][k][l] /= 255.0;

                    if (testing_data[i][j][k][l] < 0 || testing_data[i][j][k][l] > 1.0){
                        throw std::runtime_error("Error: Data Normalization Failed");
                    }
                }
            }
        }
    } 

    /* cv::Mat image = OpenCvIntegration::get_open_cv_gray_mat<double>(training_data[1011]);

    if (image.empty()) {
        std::cerr << "Error: Unable to open image!" << std::endl;
        return -1;
    }

    cv::imshow("Current Image", image);
    cv::waitKey(0); // Wait for a key press before closing the image window */

    Mat2d<double> predicted_labels;

    for (size_t i = 0; i < testing_data.size(); i++)
    {
        predicted_labels.push_back(cnn.predict_output_vector<double>(testing_data[i]));
    }
    
    DataAnalysis::display_shape(predicted_labels);

    auto true_labels = DataAnalysis::one_hot_label_encoding<double>("/home/muzaodamassa/Downloads/test_set");
    DataAnalysis::display_shape(true_labels);
    
    return 0;
}

// Proof of concept testing (Kaggle Datasets, Cats vs Dogs)
int main13()
{
    auto params = OpenCvIntegration::TrainingDataParameters("/home/muzaodamassa/Downloads/test_set", false, true, {28,28}, true, false, false);

    Mat4d<double> training_data = OpenCvIntegration::prepare_training_data<double>(params); 

    cout << "Input shape = ";
    cout << "(" << to_string(training_data.size()) << "," << to_string(training_data[0].size()) << ",";
    cout << to_string(training_data[0][0].size()) << "," << to_string(training_data[0][0][0].size()) << ")" << endl;

    // Data normalization, since max value is 255, all values will be between 0 and 1
    for (size_t i = 0; i < training_data.size(); i++) {
        for (size_t j = 0; j < training_data[i].size(); j++) {
            for (size_t k = 0; k < training_data[i][j].size(); k++) {
                for (size_t l = 0; l < training_data[i][j][k].size(); l++) {
                    training_data[i][j][k][l] /= 255.0;
                    if (training_data[i][j][k][l] < 0 || training_data[i][j][k][l] > 1.0){
                        throw std::runtime_error("Error: Data Normalization Failed");
                    }
                }
            }
        }
    } 

    NeuralNetwork model;

    auto true_labels = DataAnalysis::one_hot_label_encoding<double>("/home/muzaodamassa/Downloads/test_set");

    DataAnalysis::display_shape(true_labels);

    model.add_layer(new Conv2D<Mat4d<double>, Mat4d<double>, double>(8, 3, RELU, SAME, true));
    model.add_layer(new AveragePooling2d<Mat4d<double>, Mat4d<double>, double>(2,2));
    model.add_layer(new Conv2D<Mat4d<double>, Mat4d<double>, double>(16, 3, RELU, SAME));
    model.add_layer(new AveragePooling2d<Mat4d<double>, Mat4d<double>, double>(2,2));
    model.add_layer(new Flatten<Mat4d<double>, Mat2d<double>, double>());
    model.add_layer(new Dense<Mat2d<double>, Mat2d<double>, double>(128, RELU));
    //model.add_layer(new Dense<Mat2d<double>, Mat2d<double>, double>(32, RELU));
    model.add_layer(new Dense<Mat2d<double>, Mat2d<double>, double>(2, SOFTMAX));

    auto t1 = startBenchmark();
    Mat2d<double> result = model.fit<double>(training_data, true_labels, 2, 15, 1.0E-5);
    auto t2 = stopBenchmark();

    cout << "================================" << endl;
    cout << getDuration(t1, t2, Seconds) << endl;
    cout << "================================" << endl;

    auto shape = NumPP::get_shape(result);

    cout << "Output layer shape = ";
    cout << "(" << shape.first << ", " << shape.second << ")" << endl; 

    return 0;

    /*for (size_t i = 0; i < result.size(); i++) {
        cout << i << ", ";
        for (size_t j = 0; j < result[i].size(); j++) {
            cout << std::to_string(result[i][j]) << ", ";
        }
        cout << endl;
    } 
     */
    
    auto acc_training = model.get_accuracy<double>(result, true_labels);

    std::cout << "Predictions Accuracy (Training Dataset using result of fit method): " << acc_training*100 << "%" << std::endl;

    auto testing_params = OpenCvIntegration::TrainingDataParameters("/home/muzaodamassa/Downloads/test_set", false, true, {32,32}, true, false, false);

    Mat4d<double> testing_data = OpenCvIntegration::prepare_training_data<double>(testing_params); 

    cout << "Testing Input shape = ";
    cout << "(" << to_string(testing_data.size()) << "," << to_string(testing_data[0].size()) << ",";
    cout << to_string(testing_data[0][0].size()) << "," << to_string(testing_data[0][0][0].size()) << ")" << endl;

    // Data normalization, since max value is 255, all values will be between 0 and 1
    for (size_t i = 0; i < testing_data.size(); i++) {
        for (size_t j = 0; j < testing_data[i].size(); j++) {
            for (size_t k = 0; k < testing_data[i][j].size(); k++) {
                for (size_t l = 0; l < testing_data[i][j][k].size(); l++) {
                    testing_data[i][j][k][l] /= 255.0;
                    if (testing_data[i][j][k][l] < 0 || testing_data[i][j][k][l] > 1.0){
                        throw std::runtime_error("Error: Data Normalization Failed");
                    }
                }
            }
        }
    } 

    auto testing_labels = DataAnalysis::one_hot_label_encoding<double>("/home/muzaodamassa/Downloads/test_set");
    DataAnalysis::display_shape(testing_labels);

    Mat2d<double> predicted_labels;

    for (size_t i = 0; i < testing_labels.size(); i++)
    {
        predicted_labels.push_back(model.predict_output_vector<double>(testing_data[i]));
    }

    auto acc_testing = model.get_accuracy<double>(predicted_labels, testing_labels);

    std::cout << "Predictions Accuracy (Testing Dataset using predict method): " << acc_testing*100 << "%" << std::endl;

    //string fileName = "cats_and_dogs_model_test.bin";
    //model.export_model(fileName);

    return 0;
    
    /* Halt Execution
    char input;
    std::cout << "Program is running. Press 'q' to quit, or any other key to continue." << endl;
    
    while (true) {
        input = cin.get();

        if (input == 'q' || input == 'Q') {
            cout << "Program ended" << endl;
            return 0;
        }
        if (input != 'q' || input != 'Q') {
            break;
        }
    }
    
    cout << "Program resumed" << endl;
    return 0;
    */
}

// Mini tests for CUDA integration
/* #ifdef CUDA_AVAILABLE
    #include <cuda_runtime.h>
    extern "C" void launch_vector_add(double* A, double* B, double* C, int numElements);
    extern "C" void launch_mat_mul(double* A, double* B, double* C, int M, int N, int K);

    void queryDeviceCapabilities() {
        int device;
        cudaGetDevice(&device);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        std::cout << "Device Name: " << prop.name << std::endl;
        std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max Threads Dimension (x, y, z): (" 
                << prop.maxThreadsDim[0] << ", " 
                << prop.maxThreadsDim[1] << ", " 
                << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "Max Grid Size (x, y, z): (" 
                << prop.maxGridSize[0] << ", " 
                << prop.maxGridSize[1] << ", " 
                << prop.maxGridSize[2] << ")" << std::endl;
        std::cout << "Warp Size: " << prop.warpSize << std::endl;
        std::cout << "Total Global Memory: " << prop.totalGlobalMem << " bytes" << std::endl;
        std::cout << "Shared Memory Per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
        std::cout << "Registers Per Block: " << prop.regsPerBlock << std::endl;
        std::cout << "Memory Clock Rate: " << prop.memoryClockRate << " kHz" << std::endl;
        std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    }
#endif

// TESTING ONLY
void matMulCPU(const double* A, const double* B, double* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            C[i * K + j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i * K + j] += A[i * N + k] * B[k * K + j];
            }
        }
    }
}

bool compareResults(const double* C_gpu, const double* C_cpu, int size, double epsilon = 1e-5) {
    for (int i = 0; i < size; ++i) {
        if (fabs(C_gpu[i] - C_cpu[i]) > epsilon) {
            std::cout << "Mismatch at index " << i << ": GPU result = " << C_gpu[i] 
                      << ", CPU result = " << C_cpu[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main12() 
{
    // Vector Addition
    const int numElements = 100000;
    double* A = new double[numElements];
    double* B = new double[numElements];
    double* C = new double[numElements];

    // Matrix Multiplication
    const int M = 40000;  // Number of rows in A and C
    const int N = 40000;  // Number of columns in A and rows in B
    const int K = 40000;  // Number of columns in B and C

    double* matA = new double[M * N];
    double* matB = new double[N * K];
    double* matC_gpu = new double[M * K];
    double* matC_cpu = new double[M * K];


    // Initialize matrices with random values
    for (int i = 0; i < M * N; ++i) {
        matA[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < N * K; ++i) {
        matB[i] = static_cast<double>(rand()) / RAND_MAX;
    }

#ifdef CUDA_AVAILABLE
    std::cout << "CUDA is available. Using GPU." << std::endl;
    queryDeviceCapabilities();
    // Use CUDA function
    auto t1 = startBenchmark();
    launch_vector_add(A, B, C, numElements);
    auto t2 = stopBenchmark();
    std::cout << "================================" << std::endl;
    std::cout << "GPU Vector Addition Time: " << getDuration(t1, t2, Seconds) << " seconds" << std::endl;
    std::cout << "================================" << std::endl;
    auto t3 = startBenchmark();
    launch_mat_mul(matA, matB, matC_gpu, M, N, K);
    auto t4 = stopBenchmark();
    std::cout << "================================" << std::endl;
    std::cout << "GPU Matrix Multiplication Time: " << getDuration(t3, t4, Seconds) << " seconds" << std::endl;
    std::cout << "================================" << std::endl;   
#endif
    std::cout << "Using CPU to benchmark and make sure CUDA is working Properly" << std::endl;

    // Use CPU function
    // Rest of your main function
    auto a = NumPP::zeros<float>(100000);
    auto b = NumPP::zeros<float>(100000);

    for (int i = 0; i < a.size(); ++i)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
    }

    auto t5 = startBenchmark();
    auto c = NumPP::add<float>(a, b);
    auto t6 = stopBenchmark();

    for (size_t i = 0; i < c.size(); i++)
    {
        if (c[i] != a[i] + b[i]) {
            std::cerr << "Result verification failed at element " << i << std::endl;
            break;
        }
    }
    
    std::cout << "================================" << std::endl;
    std::cout << "CPU Vector Addition Time: " << getDuration(t5, t6, Seconds) << " seconds" << std::endl;
    std::cout << "================================" << std::endl;

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] matA;
    delete[] matB;
    delete[] matC_gpu;
    delete[] matC_cpu;
    
    return 0;
}
 */

