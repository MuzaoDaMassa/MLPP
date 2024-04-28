#include "../src/mlpp.hpp"
#include "catch.hpp"
#include "testUtils.hpp"

using namespace MLPP; 
using namespace Utils;
using namespace Benchmark;


TEST_CASE("OPEN_CV_INTEGRATION_METHODS")
{
    SECTION("CONVERT IMAGE METHOD")
    {
        SECTION("FULL SIZE COLOR IMAGE CONVERSION")
        {
            cv::Mat* imagePtr1 = new cv::Mat(cv::imread("../tests/Images/Color_Black.jpg"));
            cv::Mat* imagePtr2 = new cv::Mat(cv::imread("../tests/Images/Color_White.jpg"));
            cv::Mat* imagePtr3 = new cv::Mat(cv::imread("../tests/Images/Pista1.jpg"));

            REQUIRE(imagePtr1->data);
            REQUIRE(imagePtr2->data);
            REQUIRE(imagePtr3->data);

            auto cvt_ImagePtr1 = OpencvIntegration::convert_image(imagePtr1);
            auto cvt_ImagePtr2 = OpencvIntegration::convert_image(imagePtr2);
            auto cvt_ImagePtr3 = OpencvIntegration::convert_image(imagePtr3);

            for (size_t i = 0; i < imagePtr1->rows; i++) {
                for (size_t j = 0; j < imagePtr1->cols; j++) {
                    cv::Vec3b pixel = imagePtr1->at<cv::Vec3b>(i, j);
                    for (size_t k = 0; k < 3; k++) {
                        REQUIRE((*cvt_ImagePtr1)[i][j][k] == pixel[k]);
                    }
                }
            }

            for (size_t i = 0; i < imagePtr2->rows; i++) {
                for (size_t j = 0; j < imagePtr2->cols; j++) {
                    cv::Vec3b pixel = imagePtr2->at<cv::Vec3b>(i, j);
                    for (size_t k = 0; k < 3; k++) {
                        REQUIRE((*cvt_ImagePtr2)[i][j][k] == pixel[k]);
                    }
                }
            }

            for (size_t i = 0; i < imagePtr3->rows; i++) {
                for (size_t j = 0; j < imagePtr3->cols; j++) {
                    cv::Vec3b pixel = imagePtr3->at<cv::Vec3b>(i, j);
                    for (size_t k = 0; k < 3; k++) {
                        REQUIRE((*cvt_ImagePtr3)[i][j][k] == pixel[k]);
                    }
                }
            }

            delete imagePtr1, imagePtr2, imagePtr3;
            delete cvt_ImagePtr1, cvt_ImagePtr2, cvt_ImagePtr3;
        }

        SECTION("FULL SIZE GRAY IMAGE CONVERSION")
        {
            cv::Mat* imagePtr1 = new cv::Mat(cv::imread("../tests/Images/Color_White.jpg"));
            cv::Mat* imagePtr2 = new cv::Mat(cv::imread("../tests/Images/Color_Black.jpg"));
            cv::Mat* imagePtr3 = new cv::Mat(cv::imread("../tests/Images/Pista1.jpg"));

            REQUIRE(imagePtr1->data);
            REQUIRE(imagePtr2->data);
            REQUIRE(imagePtr3->data);

            cv::Mat grayImage1;
            cv::cvtColor(*imagePtr1, grayImage1, cv::COLOR_BGR2GRAY);
            cv::Mat grayImage2;
            cv::cvtColor(*imagePtr2, grayImage2, cv::COLOR_BGR2GRAY);
            cv::Mat grayImage3;
            cv::cvtColor(*imagePtr3, grayImage3, cv::COLOR_BGR2GRAY);
            cv::Mat* grayPtr1 = new cv::Mat(grayImage1);
            cv::Mat* grayPtr2 = new cv::Mat(grayImage2);
            cv::Mat* grayPtr3 = new cv::Mat(grayImage3);

            REQUIRE(grayPtr1->data);
            REQUIRE(grayPtr2->data);
            REQUIRE(grayPtr3->data);

            auto cvt_ImagePtr1 = OpencvIntegration::convert_image(grayPtr1);
            auto cvt_ImagePtr2 = OpencvIntegration::convert_image(grayPtr2);
            auto cvt_ImagePtr3 = OpencvIntegration::convert_image(grayPtr3);

            for (size_t i = 0; i < grayPtr1->rows; i++) {
                for (size_t j = 0; j < grayPtr1->cols; j++) {
                    cv::Vec3b pixel = grayPtr1->at<cv::Vec3b>(i, j);
                    for (size_t k = 0; k < 3; k++) {
                        REQUIRE((*cvt_ImagePtr1)[i][j][k] == pixel[k]);
                    }
                }
            }

            for (size_t i = 0; i < grayPtr2->rows; i++) {
                for (size_t j = 0; j < grayPtr2->cols; j++) {
                    cv::Vec3b pixel = grayPtr2->at<cv::Vec3b>(i, j);
                    for (size_t k = 0; k < 3; k++) {
                        REQUIRE((*cvt_ImagePtr2)[i][j][k] == pixel[k]);
                    }
                }
            }

            for (size_t i = 0; i < grayPtr3->rows; i++) {
                for (size_t j = 0; j < grayPtr3->cols; j++) {
                    cv::Vec3b pixel = grayPtr3->at<cv::Vec3b>(i, j);
                    for (size_t k = 0; k < 3; k++) {
                        REQUIRE((*cvt_ImagePtr3)[i][j][k] == pixel[k]);
                    }
                }
            }

            delete imagePtr1, imagePtr2, imagePtr3;
            delete grayPtr1, grayPtr2, grayPtr3;
            delete cvt_ImagePtr1, cvt_ImagePtr2, cvt_ImagePtr3;
        }

        SECTION("RESIZED COLOR IMAGE CONVERSION")
        {
            cv::Mat* imagePtr1 = new cv::Mat(cv::imread("../tests/Images/Color_Black.jpg"));
            cv::Mat* imagePtr2 = new cv::Mat(cv::imread("../tests/Images/Color_White.jpg"));
            cv::Mat* imagePtr3 = new cv::Mat(cv::imread("../tests/Images/Pista1.jpg"));

            REQUIRE(imagePtr1->data);
            REQUIRE(imagePtr2->data);
            REQUIRE(imagePtr3->data);

            cv::Mat resizedImage1;  
            cv::resize(*imagePtr1, resizedImage1, cv::Size(420, 240));
            cv::Mat resizedImage2;  
            cv::resize(*imagePtr2, resizedImage2, cv::Size(420, 240));
            cv::Mat resizedImage3;  
            cv::resize(*imagePtr3, resizedImage3, cv::Size(420, 240));
            cv::Mat* rImagePtr1 = new cv::Mat(resizedImage1);
            cv::Mat* rImagePtr2 = new cv::Mat(resizedImage2);
            cv::Mat* rImagePtr3 = new cv::Mat(resizedImage3);

            REQUIRE(rImagePtr1->data);
            REQUIRE(rImagePtr2->data);
            REQUIRE(rImagePtr3->data);
            
            auto cvt_ImagePtr1 = OpencvIntegration::convert_image(rImagePtr1);
            auto cvt_ImagePtr2 = OpencvIntegration::convert_image(rImagePtr2);
            auto cvt_ImagePtr3 = OpencvIntegration::convert_image(rImagePtr3);

            for (size_t i = 0; i < rImagePtr1->rows; i++) {
                for (size_t j = 0; j < rImagePtr1->cols; j++) {
                    cv::Vec3b pixel = rImagePtr1->at<cv::Vec3b>(i, j);
                    for (size_t k = 0; k < 3; k++) {
                        REQUIRE((*cvt_ImagePtr1)[i][j][k] == pixel[k]);
                    }
                }
            }

            for (size_t i = 0; i < rImagePtr2->rows; i++) {
                for (size_t j = 0; j < rImagePtr2->cols; j++) {
                    cv::Vec3b pixel = rImagePtr2->at<cv::Vec3b>(i, j);
                    for (size_t k = 0; k < 3; k++) {
                        REQUIRE((*cvt_ImagePtr2)[i][j][k] == pixel[k]);
                    }
                }
            }

            for (size_t i = 0; i < rImagePtr3->rows; i++) {
                for (size_t j = 0; j < rImagePtr3->cols; j++) {
                    cv::Vec3b pixel = rImagePtr3->at<cv::Vec3b>(i, j);
                    for (size_t k = 0; k < 3; k++) {
                        REQUIRE((*cvt_ImagePtr3)[i][j][k] == pixel[k]);
                    }
                }
            }

            delete imagePtr1, imagePtr2, imagePtr3;
            delete rImagePtr1, rImagePtr2, rImagePtr3;
            delete cvt_ImagePtr1, cvt_ImagePtr2, cvt_ImagePtr3;
        }
    
        SECTION("RESIZED GRAY IMAGE CONVERSION")
        {
            cv::Mat* imagePtr1 = new cv::Mat(cv::imread("../tests/Images/Color_White.jpg"));
            cv::Mat* imagePtr2 = new cv::Mat(cv::imread("../tests/Images/Color_Black.jpg"));
            cv::Mat* imagePtr3 = new cv::Mat(cv::imread("../tests/Images/Pista1.jpg"));

            REQUIRE(imagePtr1->data);
            REQUIRE(imagePtr2->data);
            REQUIRE(imagePtr3->data);

            cv::Mat resizedImage1;  
            cv::resize(*imagePtr1, resizedImage1, cv::Size(420, 240));
            cv::Mat resizedImage2;  
            cv::resize(*imagePtr2, resizedImage2, cv::Size(420, 240));
            cv::Mat resizedImage3;  
            cv::resize(*imagePtr3, resizedImage3, cv::Size(420, 240));
            cv::Mat* rImagePtr1 = new cv::Mat(resizedImage1);
            cv::Mat* rImagePtr2 = new cv::Mat(resizedImage2);
            cv::Mat* rImagePtr3 = new cv::Mat(resizedImage3);

            REQUIRE(rImagePtr1->data);
            REQUIRE(rImagePtr2->data);
            REQUIRE(rImagePtr3->data);

            cv::Mat grayImage1;
            cv::cvtColor(*rImagePtr1, grayImage1, cv::COLOR_BGR2GRAY);
            cv::Mat grayImage2;
            cv::cvtColor(*rImagePtr2, grayImage2, cv::COLOR_BGR2GRAY);
            cv::Mat grayImage3;
            cv::cvtColor(*rImagePtr3, grayImage3, cv::COLOR_BGR2GRAY);
            cv::Mat* grayPtr1 = new cv::Mat(grayImage1);
            cv::Mat* grayPtr2 = new cv::Mat(grayImage2);
            cv::Mat* grayPtr3 = new cv::Mat(grayImage3);

            REQUIRE(grayPtr1->data);
            REQUIRE(grayPtr2->data);
            REQUIRE(grayPtr3->data);

            auto cvt_ImagePtr1 = OpencvIntegration::convert_image(grayPtr1);
            auto cvt_ImagePtr2 = OpencvIntegration::convert_image(grayPtr2);
            auto cvt_ImagePtr3 = OpencvIntegration::convert_image(grayPtr3);

            for (size_t i = 0; i < grayPtr1->rows; i++) {
                for (size_t j = 0; j < grayPtr1->cols; j++) {
                    cv::Vec3b pixel = grayPtr1->at<cv::Vec3b>(i, j);
                    for (size_t k = 0; k < 3; k++) {
                        REQUIRE((*cvt_ImagePtr1)[i][j][k] == pixel[k]);
                    }
                }
            }

            for (size_t i = 0; i < grayPtr2->rows; i++) {
                for (size_t j = 0; j < grayPtr2->cols; j++) {
                    cv::Vec3b pixel = grayPtr2->at<cv::Vec3b>(i, j);
                    for (size_t k = 0; k < 3; k++) {
                        REQUIRE((*cvt_ImagePtr2)[i][j][k] == pixel[k]);
                    }
                }
            }

            for (size_t i = 0; i < grayPtr3->rows; i++) {
                for (size_t j = 0; j < grayPtr3->cols; j++) {
                    cv::Vec3b pixel = grayPtr3->at<cv::Vec3b>(i, j);
                    for (size_t k = 0; k < 3; k++) {
                        REQUIRE((*cvt_ImagePtr3)[i][j][k] == pixel[k]);
                    }
                }
            }

            delete imagePtr1, imagePtr2, imagePtr3;
            delete grayPtr1, grayPtr2, grayPtr3;
            delete rImagePtr1, rImagePtr2, rImagePtr3;
            delete cvt_ImagePtr1, cvt_ImagePtr2, cvt_ImagePtr3;
        }
    }
}


TEST_CASE("OPEN_CV_INTEGRATION_METHODS_BENCHHMARKS", "[.OpenCV_Integration_Benchmarks]")
{
    SECTION("CONVERT IMAGE METHOD")
    {
        SECTION("FULL SIZE COLOR IMAGE CONVERSION - ALL BLACK JPG BENCHMARK")
        {
            cv::Mat* imagePtr = new cv::Mat(cv::imread("../tests/Images/Color_Black.jpg"));

            auto start = startBenchmark();
            auto cvt_ImagePtr = OpencvIntegration::convert_image(imagePtr);
            auto stop = stopBenchmark();

            std::cout << "FULL SIZE COLOR IMAGE CONVERSION - ALL BLACK JPG BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Seconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;

            delete imagePtr;
            delete cvt_ImagePtr;
        }

        SECTION("FULL SIZE COLOR IMAGE CONVERSION - ALL WHITE JPG BENCHMARK")
        {
            cv::Mat* imagePtr = new cv::Mat(cv::imread("../tests/Images/Color_White.jpg"));

            auto start = startBenchmark();
            auto cvt_ImagePtr = OpencvIntegration::convert_image(imagePtr);
            auto stop = stopBenchmark();

            std::cout << "FULL SIZE COLOR IMAGE CONVERSION - ALL WHITE JPG BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Seconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;

            delete imagePtr;
            delete cvt_ImagePtr;
        }

        SECTION("FULL SIZE COLOR IMAGE CONVERSION - TEST IMAGE JPG BENCHMARK")
        {
            cv::Mat* imagePtr = new cv::Mat(cv::imread("../tests/Images/Pista1.jpg"));

            auto start = startBenchmark();
            auto cvt_ImagePtr = OpencvIntegration::convert_image(imagePtr);
            auto stop = stopBenchmark();

            std::cout << "FULL SIZE COLOR IMAGE CONVERSION - TEST IMAGE JPG BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Seconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;

            delete imagePtr;
            delete cvt_ImagePtr;
        }

        SECTION("FULL SIZE GRAY IMAGE CONVERSION - ALL BLACK JPG BENCHMARK")
        {
            cv::Mat* imagePtr = new cv::Mat(cv::imread("../tests/Images/Color_Black.jpg"));
            cv::Mat grayImage;
            cv::cvtColor(*imagePtr, grayImage, cv::COLOR_BGR2GRAY);
            cv::Mat* grayPtr = new cv::Mat(grayImage);

            auto start = startBenchmark();
            auto cvt_ImagePtr = OpencvIntegration::convert_image(grayPtr);
            auto stop = stopBenchmark();

            std::cout << "FULL SIZE GRAY IMAGE CONVERSION - ALL BLACK JPG BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Seconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;

            delete imagePtr;
            delete grayPtr;
            delete cvt_ImagePtr;
        }

        SECTION("FULL SIZE GRAY IMAGE CONVERSION - ALL WHITE JPG BENCHMARK")
        {
            cv::Mat* imagePtr = new cv::Mat(cv::imread("../tests/Images/Color_White.jpg"));
            cv::Mat grayImage;
            cv::cvtColor(*imagePtr, grayImage, cv::COLOR_BGR2GRAY);
            cv::Mat* grayPtr = new cv::Mat(grayImage);

            auto start = startBenchmark();
            auto cvt_ImagePtr = OpencvIntegration::convert_image(grayPtr);
            auto stop = stopBenchmark();

            std::cout << "FULL SIZE GRAY IMAGE CONVERSION - ALL WHITE JPG BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Seconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;

            delete imagePtr;
            delete grayPtr;
            delete cvt_ImagePtr;
        }

        SECTION("FULL SIZE GRAY IMAGE CONVERSION - TEST IMAGE JPG BENCHMARK")
        {
            cv::Mat* imagePtr = new cv::Mat(cv::imread("../tests/Images/Pista1.jpg"));
            cv::Mat grayImage;
            cv::cvtColor(*imagePtr, grayImage, cv::COLOR_BGR2GRAY);
            cv::Mat* grayPtr = new cv::Mat(grayImage);

            auto start = startBenchmark();
            auto cvt_ImagePtr = OpencvIntegration::convert_image(grayPtr);
            auto stop = stopBenchmark();

            std::cout << "FULL SIZE GRAY IMAGE CONVERSION - TEST IMAGE JPG BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Seconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;

            delete imagePtr;
            delete grayPtr;
            delete cvt_ImagePtr;
        }

        SECTION("RESIZED COLOR IMAGE CONVERSION - ALL BLACK JPG")
        {
            cv::Mat* imagePtr = new cv::Mat(cv::imread("../tests/Images/Color_Black.jpg"));
            cv::Mat resizedImage;  
            cv::resize(*imagePtr, resizedImage, cv::Size(420, 240));
            cv::Mat* rImagePtr = new cv::Mat(resizedImage);
            
            auto start = startBenchmark();
            auto cvt_ImagePtr = OpencvIntegration::convert_image(rImagePtr);
            auto stop = stopBenchmark();

            std::cout << "RESIZED COLOR IMAGE CONVERSION - ALL BLACK JPG BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Seconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;

            delete imagePtr;
            delete rImagePtr;
            delete cvt_ImagePtr;
        }

        SECTION("RESIZED COLOR IMAGE CONVERSION - ALL WHITE JPG")
        {
            cv::Mat* imagePtr = new cv::Mat(cv::imread("../tests/Images/Color_White.jpg"));
            cv::Mat resizedImage;  
            cv::resize(*imagePtr, resizedImage, cv::Size(420, 240));
            cv::Mat* rImagePtr = new cv::Mat(resizedImage);
            
            auto start = startBenchmark();
            auto cvt_ImagePtr = OpencvIntegration::convert_image(rImagePtr);
            auto stop = stopBenchmark();

            std::cout << "RESIZED COLOR IMAGE CONVERSION - ALL WHITE JPG BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Seconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;

            delete imagePtr;
            delete rImagePtr;
            delete cvt_ImagePtr;
        }

        SECTION("RESIZED COLOR IMAGE CONVERSION - TEST IMAGE JPG")
        {
            cv::Mat* imagePtr = new cv::Mat(cv::imread("../tests/Images/Pista1.jpg"));
            cv::Mat resizedImage;  
            cv::resize(*imagePtr, resizedImage, cv::Size(420, 240));
            cv::Mat* rImagePtr = new cv::Mat(resizedImage);
            
            auto start = startBenchmark();
            auto cvt_ImagePtr = OpencvIntegration::convert_image(rImagePtr);
            auto stop = stopBenchmark();

            std::cout << "RESIZED COLOR IMAGE CONVERSION - TEST IMAGE JPG BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Seconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;

            delete imagePtr;
            delete rImagePtr;
            delete cvt_ImagePtr;
        }

        SECTION("RESIZED GRAY IMAGE CONVERSION - ALL BLACK JPG")
        {
            cv::Mat* imagePtr = new cv::Mat(cv::imread("../tests/Images/Color_Black.jpg"));
            cv::Mat resizedImage;  
            cv::resize(*imagePtr, resizedImage, cv::Size(420, 240));
            cv::Mat* rImagePtr = new cv::Mat(resizedImage);
            cv::Mat grayImage;
            cv::cvtColor(*rImagePtr, grayImage, cv::COLOR_BGR2GRAY);
            cv::Mat* grayPtr = new cv::Mat(grayImage);
            
            auto start = startBenchmark();
            auto cvt_ImagePtr = OpencvIntegration::convert_image(grayPtr);
            auto stop = stopBenchmark();

            std::cout << "RESIZED GRAY IMAGE CONVERSION - ALL BLACK JPG BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Seconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;

            delete imagePtr;
            delete rImagePtr;
            delete cvt_ImagePtr;
        }

        SECTION("RESIZED GRAY IMAGE CONVERSION - ALL WHITE JPG")
        {
            cv::Mat* imagePtr = new cv::Mat(cv::imread("../tests/Images/Color_White.jpg"));
            cv::Mat resizedImage;  
            cv::resize(*imagePtr, resizedImage, cv::Size(420, 240));
            cv::Mat* rImagePtr = new cv::Mat(resizedImage);
            cv::Mat grayImage;
            cv::cvtColor(*rImagePtr, grayImage, cv::COLOR_BGR2GRAY);
            cv::Mat* grayPtr = new cv::Mat(grayImage);
            
            auto start = startBenchmark();
            auto cvt_ImagePtr = OpencvIntegration::convert_image(grayPtr);
            auto stop = stopBenchmark();

            std::cout << "RESIZED GRAY IMAGE CONVERSION - ALL WHITE JPG BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Seconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;

            delete imagePtr;
            delete rImagePtr;
            delete cvt_ImagePtr;
        }
        
        SECTION("RESIZED GRAY IMAGE CONVERSION - TEST IMAGE JPG")
        {
            cv::Mat* imagePtr = new cv::Mat(cv::imread("../tests/Images/Pista1.jpg"));
            cv::Mat resizedImage;  
            cv::resize(*imagePtr, resizedImage, cv::Size(420, 240));
            cv::Mat* rImagePtr = new cv::Mat(resizedImage);
            cv::Mat grayImage;
            cv::cvtColor(*rImagePtr, grayImage, cv::COLOR_BGR2GRAY);
            cv::Mat* grayPtr = new cv::Mat(grayImage);
            
            auto start = startBenchmark();
            auto cvt_ImagePtr = OpencvIntegration::convert_image(grayPtr);
            auto stop = stopBenchmark();

            std::cout << "RESIZED GRAY IMAGE CONVERSION - TEST IMAGE JPG BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Seconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;

            delete imagePtr;
            delete rImagePtr;
            delete cvt_ImagePtr;
        }
    }
}
