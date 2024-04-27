#include <iostream>
#include "mlpp.hpp"

using namespace std;
using namespace MLPP;


// Mini tests for NumPP unit
int main()
{
    auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/hw_100.csv");
    //auto data2 = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/hw_100.csv");
    auto cData = DataAnalysis::matrix_converter<double>(data);    
    //auto cData2 = DataAnalysis::matrix_converter<int>(data2);

    Mat2d<int> original = {{1, 2}, {3, 4}, {5, 6}};


    Mat2d<float> a = {{0.1, 0.1, 0.1, 0.1}, {0.2, 0.2, 0.2, 0.2}, {0.3, 0.3, 0.3, 0.3}, {0.4, 0.4, 0.4, 0.4}};
    Mat2d<float> b = {{0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}};

    //Mat2d<float> r = NumPP::rand<float>(20, 15, 20.0, 5.0);
    Mat2d<int> T = NumPP::transpose(original);
    Mat2d<int> r1 = NumPP::dot(T, original);


    DataAnalysis::displayAll(r1);
}

// Mini tests for data analysis unit
int main1() 
{ 
    auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
    auto data2 = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/organizations-100.csv");
    auto cData = DataAnalysis::matrix_converter<float>(data);
    // Print data
    cout << "========================================" << endl;
    //DataAnalysis::displayAll(data);
    cout << "========================================" << endl;
    vector<int> rows{0, 10, 20, 30, 40, 50, 60, 70, 80, 90};
    //DataAnalysis::displayRows(data, rows);
    cout << "========================================" << endl;
    vector<int> cols{2, 4, 6, 8};
    //DataAnalysis::displayColumns(data, cols);
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
    DataAnalysis::displayRows(data, rows);
    //DataAnalysis::displayColumns(data, cols);
    // cout << el << endl;
    // cout << pos[0] << ", " << pos[1] << endl;
    //DataAnalysis::displayAll(data);
    DataAnalysis::displayHead(data, 20);
    //DataAnalysis::displayBottom(data, 5);

    return 0;
}

/* // Open CV testing
int main2()
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

		// If the frame is empty, break immediately
		if (frame.empty())
		{
			break;
		}

        std::cout << frame.type() << std::endl;

		// Convert the frame to grayscale
		cv::Mat gray;
		cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::cout << gray.type() << std::endl;

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
        break;
    }

	// Whene everything is done, release the video capture and video writer objects
	cap.release();

    // Close all windows
	cv::destroyAllWindows();

    return 0;	
}
 */