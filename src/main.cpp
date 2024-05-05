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
int main3()
{
    cv::Mat* imagePtr = new cv::Mat(cv::imread("../tests/Images/Pista1.jpg"));
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
    auto nImagePtr = OpencvIntegration::convert_image(imagePtr);
    auto end = stopBenchmark();

    cout << getDuration(start, end, Seconds) << endl;

    // Access the element at row 'i' and column 'j'
    std::vector<uint8_t>& pixel = (*nImagePtr)[150][150];
    cv::Vec3b& cvPixel = imagePtr->at<cv::Vec3b>(150, 150);
    // Now you can access individual channels of the pixel
    uint8_t blue = pixel[0];
    uint8_t green = pixel[1];
    uint8_t red = pixel[2];
    
    cout << cvPixel << endl;
    cout << "[" << to_string(pixel[0]) << ", " << to_string(pixel[1]) << ", " << to_string(pixel[2]) << "]" << endl;
    cout << endl;

    delete imagePtr;
    //delete rImagePtr;
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

        auto matPtr = OpencvIntegration::convert_image(framePtr);

        // Access the element at row 'i' and column 'j'
        std::vector<uint8_t> &pixel = (*matPtr)[150][150];

        // Now you can access individual channels of the pixel
        uint8_t blue = pixel[0];
        uint8_t green = pixel[1];
        uint8_t red = pixel[2];
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

// Mini tests for Neural Network unit
int main()
{
    cv::Mat* imagePtr = new cv::Mat(cv::imread("../tests/Images/Faixa2.jpg"));
	if (!imagePtr->data) { 
		std::cerr << "Error: No image data" << std::endl; 
		return -1; 
	}  

    cv::Mat resizedImage;  
    cv::resize(*imagePtr, resizedImage, cv::Size(420, 240));
    cv::Mat* rImagePtr = new cv::Mat(resizedImage);

    //cv::Mat grayImage;
    //cv::cvtColor(*rImagePtr, grayImage, cv::COLOR_BGR2GRAY);
    //cv::Mat* grayPtr = new cv::Mat(grayImage);

    //auto nImagePtr = OpencvIntegration::convert_image(imagePtr);
    auto matPtr = OpencvIntegration::get_sum_pixels(rImagePtr);

    // Example usage of neural networ class
    //Mat2d<double> x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    Mat2d<double> x = DataAnalysis::matrix_converter<int16_t, double>(*matPtr);
    Mat2d<double> y = NumPP::zeros<double>(240, 1);

    int center_y = x[0].size() /2;
    double threshold = 600;

    for (size_t i = 0; i < x.size(); i++) {
        for (size_t j = 0; j < x[i].size(); j++) {
            if (j < center_y) {
               if (x[i][j] >= threshold) {
                y[i][0] = -1.0;
               }
            }
            else {
               if (x[i][j] >= threshold) {
                y[i][0] = 1.0;
               }
            }
        }
    }

    int input_size = x[0].size();
    //int hidden_size1 = input_size/2;
    //int hidden_size2 = input_size/4;

    // Define neural network with specified architecture
    NeuralNetwork model(input_size, 120, 60, 1);

    // Train neural network with specified training data and hyperparameters
    size_t epochs = 1100;
    double learning_rate = 0.1;
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        // Forward propagation
        Mat2d<double> output = model.forward(x);

        // Compute loss
        double loss = 0.0;
        for (size_t i = 0; i < output.size(); i++) {
            loss += pow(output[i][0] - y[i][0], 2);
        }

        loss /= y.size();

        // Print loss periodically
        if (epoch % 100 == 0) {
            cout << "Epoch " << epoch << ", Loss: " << loss << endl;
        }

        // Backward propagation
        model.backward(x, y, learning_rate);
    }
    return 0;

}