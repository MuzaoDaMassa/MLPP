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
    auto data2 = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/organizations-100.csv");
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
    DataAnalysis::display_rows(data, rows);
    //DataAnalysis::display_columns(data, cols);
    // cout << el << endl;
    // cout << pos[0] << ", " << pos[1] << endl;
    //DataAnalysis::display_all(data);
    DataAnalysis::display_head(data, 20);
    //DataAnalysis::display_bottom(data, 5);

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

    Mat2d<int16_t> a {{-1,-1,-1}, {0,1,-1}, {0,1,1}};
    Mat2d<int16_t> b {{0,0,0}, {0,156,155}, {0,153,154}};
    Mat2d<int16_t>* aPtr = new Mat2d<int16_t>(a);
    Mat2d<int16_t>* bPtr = new Mat2d<int16_t>(b);
    //auto r = NumPP::sum_mat_mul_matching_elements(aPtr, bPtr);
    //Mat2d<double> r = NumPP::scalar_mat_mul<int16_t, double>(aPtr, 2.0);
    //cout << to_string(r) << endl;

    //vector<u_int8_t> v {255, 255, 255};
    //cout << to_string(NumPP::get_sum_of_vector<u_int8_t, u_int16_t>(v)) << endl;

    //DataAnalysis::display_all(r);
    std::cout << "-------------------------------------------------------------" << std::endl;
    //DataAnalysis::display_all(r1);

    return 0;
}

// Open CV Image testing
int main()
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

    cv::Mat* imagePtr = new cv::Mat(cv::imread("../tests/Images/Pista1.jpg"));
	if (!imagePtr->data) { 
		std::cerr << "Error: No image data" << std::endl; 
		return -1; 
	} 

    cv::Mat resizedImage;  
    cv::resize(*imagePtr, resizedImage, cv::Size(240, 240));

    cv::Mat grayImage;
    cv::cvtColor(resizedImage, grayImage, cv::COLOR_BGR2GRAY);

    cv::namedWindow("Gray Scale Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Gray Scale Image", grayImage);
    cv::waitKey(0);

    cv::Mat bev_image = OpencvIntegration::change_perspective_to_bev(grayImage);

    cv::namedWindow("Birds Eye View", cv::WINDOW_AUTOSIZE);
    cv::imshow("Birds Eye View", bev_image);
    cv::waitKey(0);


    return 0;
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
    cv::namedWindow("Video Stream", cv::WINDOW_NORMAL);
    
	// Set capture resolution
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 420);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);

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

        auto matPtr = OpencvIntegration::convert_color_image<float>(framePtr);

        // Access the element at row 'i' and column 'j'
        std::vector<float> &pixel = (*matPtr)[150][150];

        // Now you can access individual channels of the pixel
        float blue = pixel[0];
        float green = pixel[1];
        float red = pixel[2];
        cout << framePtr->at<cv::Vec3b>(150, 150) << endl;
        cout << "[" << to_string(pixel[0]) << ", " << to_string(pixel[1]) << ", " << to_string(pixel[2]) << "]" << endl;

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

// Mini tests for Open Cv integration + Neural network units
int main5()
{
    string dir_path = "../tests/Images";

    auto training_data = OpencvIntegration::prepare_training_data<float>(dir_path, true, true, {28, 28});
 
    cout << "Input shape = ";
    cout << "(" << to_string(training_data.size()) << "," << to_string(training_data[0].size()) << ",";
    cout << to_string(training_data[0][0].size()) << "," << to_string(training_data[0][0][0].size()) << ")" << endl;

    auto conv_output = NeuralNetwork::conv_2d(training_data, 5, 3, RELU, SAME);

    cout << "Convolution output shape = ";
    cout << "(" << to_string(conv_output.size()) << "," << to_string(conv_output[0].size()) << ",";
    cout << to_string(conv_output[0][0].size()) << "," << to_string(conv_output[0][0][0].size()) << ")" << endl;
    
    /*
    for (size_t i = 0; i < conv_output.size(); i++) {
        auto cv2Image = OpencvIntegration::get_open_cv_gray_mat<float>(&conv_output[i], 4);
        string window_name = "Training data " + to_string(i + 1);
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
        cv::imshow(window_name, cv2Image);
        cv::waitKey(0);
    } 
    */
    
    auto pool_output = NeuralNetwork::pooling(conv_output, 2, 2);

    cout << "Pooling output shape = ";
    cout << "(" << to_string(pool_output.size()) << "," << to_string(pool_output[0].size()) << ",";
    cout << to_string(pool_output[0][0].size()) << "," << to_string(pool_output[0][0][0].size()) << ")" << endl;

    /*
    for (size_t i = 0; i < pool_output.size(); i++) {
        auto cv2Image = OpencvIntegration::get_open_cv_gray_mat<float>(&pool_output[i], 4);
        string window_name = "Training data " + to_string(i + 1);
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
        cv::imshow(window_name, cv2Image);
        cv::waitKey(0);
    } 
    */
     
    auto flatten_output = NeuralNetwork::flatten(pool_output);
    auto shape = NumPP::get_shape(flatten_output);

    cout << "Flatten output shape = ";
    cout << "(" << shape.first << ", " << shape.second << ")" << endl;

    std::vector<float> bias_vector = NumPP::ones<float>(120);
    Mat2d<float> weight_matrix = NumPP::rand<float>(flatten_output[0].size(), 120, -1, 1);
    //Mat2d<float> weight_matrix = NumPP::ones<float>(flatten_output[0].size(), 120);
    auto dense_output = NeuralNetwork::dense(flatten_output, weight_matrix, bias_vector, 120, RELU);

    shape = NumPP::get_shape(dense_output);

    cout << "Dense 1 output shape = ";
    cout << "(" << shape.first << ", " << shape.second << ")" << endl;

    bias_vector = NumPP::ones<float>(3);
    weight_matrix = NumPP::rand<float>(dense_output[0].size(), 3, 0.0, 0.5);
    //weight_matrix = NumPP::ones<float>(dense_output[0].size(), 3);
    auto dense_output_2 = NeuralNetwork::dense(dense_output, weight_matrix, bias_vector, 3, SOFTMAX);
    
    shape = NumPP::get_shape(dense_output_2);

    cout << "Dense 2 output shape = ";
    cout << "(" << shape.first << ", " << shape.second << ")" << endl;
    
    for (size_t i = 0; i < dense_output_2.size(); i++) {
        cout << i << ", ";
        for (size_t j = 0; j < dense_output_2[i].size(); j++) {
            cout << dense_output_2[i][j] << ", ";
        }
        cout << endl;
    }

    Utils::check_softmax_sums(dense_output_2);
    
    return 0;
}


/*  Mini tests for Neural Network unit
int main6()
{
    cv::Mat* imagePtr = new cv::Mat(cv::imread("../tests/Images/Exterior.jpeg"));
	if (!imagePtr->data) { 
		std::cerr << "Error: No image data" << std::endl; 
		return -1; 
	}  

    cv::Mat resizedImage;  
    cv::resize(*imagePtr, resizedImage, cv::Size(1000, 1000));
    cv::Mat* rImagePtr = new cv::Mat(resizedImage);

    cv::Mat grayImage;
    cv::cvtColor(*rImagePtr, grayImage, cv::COLOR_BGR2GRAY);
    cv::Mat* grayPtr = new cv::Mat(grayImage);

    cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original Image", *grayPtr);
    cv::waitKey(0);

    Mat2d<int8_t> filter_1{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    Mat2d<int8_t> filter_2{{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};
    Mat2d<int8_t> filter_3{{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    auto nImagePtr = OpencvIntegration::convert_gray_image(grayPtr);
    //auto nImagePtr = OpencvIntegration::convert_color_image(rImagePtr);
    //auto fImage = ComputerVision::conv_2d<int16_t, u_int8_t>(*nImagePtr, filter_1, 3);
    //auto fImage = NeuralNetwork::conv_2d<u_int8_t, u_int8_t>(*nImagePtr, filter_1, NeuralNetwork::SAME, 3);
    //auto rImage = NumPP::tanh(fImage);
    //auto rImage = NumPP::relu(fImage);
    //auto cv2Image = OpencvIntegration::get_open_cv_color_mat<u_int8_t>(nImagePtr);
    auto cv2Image = OpencvIntegration::get_open_cv_gray_mat<u_int8_t>(nImagePtr);
    //auto cv2Image = OpencvIntegration::get_open_cv_gray_mat<u_int8_t>(&fImage);
    //auto cv2Image = OpencvIntegration::get_open_cv_gray_mat<u_int8_t>(&rImage);
    //auto cv2Image = OpencvIntegration::get_open_cv_color_mat<int16_t>(&fImage);
    //auto cv2Image = OpencvIntegration::get_open_cv_color_mat<u_int8_t>(&fImage);
    //auto cv2Image = OpencvIntegration::get_open_cv_color_mat<int16_t>(&rImage);

    cv::namedWindow("Convolution Result Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Convolution Result Image", cv2Image);
    cv::waitKey(0);

    //auto pImage = ComputerVision::pooling(fImage);
    //auto pImage = NeuralNetwork::pooling(rImage);
    //auto cv2Image_2 = OpencvIntegration::get_open_cv_gray_mat<u_int8_t>(&pImage);
    //auto cv2Image_2 = OpencvIntegration::get_open_cv_color_mat<int16_t>(&pImage);

    //cv::namedWindow("Pooling Result Image", cv::WINDOW_AUTOSIZE);
    //cv::imshow("Pooling Result Image", cv2Image_2);
    //cv::waitKey(0);

    return 0;
}
 */