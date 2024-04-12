#include "catch.hpp"
#include "../src/mlpp.hpp"
#include "testUtils.hpp"

using namespace Benchmark;
using namespace DataAnalysis;

TEST_CASE ("Data Analysis Unit Testing")
{
    SECTION ("Read CSV Methods")
    {
        auto data1 = readCSV<std::string>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
        auto data2 = readCSV<float>("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
        auto data3 = readCSV<double>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv"); 

        REQUIRE(data1.size() > 0); 

        for (const auto& row : data1) 
        {
            for (const auto& element : row) 
            {
                REQUIRE(!element.empty());
            }
        } 

        REQUIRE(data2.size() > 0); 

        for (const auto& row : data2) 
        {
            for (const auto& element : row) 
            {
                REQUIRE(element >= 0);
            }
        } 

        REQUIRE(data3.size() > 0);   

        for (const auto& row : data3) 
        {
            for (const auto& element : row) 
            {
                REQUIRE(element >= 0);
            }
        } 
    }
}

TEST_CASE("Data Analysis Unit Benchmarking", "[.benchmark]")
{
    SECTION("100 ROWS READ CSV BENCHMARK")
    {
        auto start = startBenchmark();
        auto data = readCSV<std::string>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
        auto stop = stopBenchmark();

        CHECK(data.size() > 0);
        //INFO("100 ROW DATABASE")
        std::cout << "100 ROW DATABASE" << std::endl;
        std::cout << getDuration(start, stop, Microseconds) << std::endl;
    }

    SECTION("10 000 ROWS READ CSV BENCHMARK")
    {
        auto start = startBenchmark();
        auto data = readCSV<std::string>("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
        auto stop = stopBenchmark();

        CHECK(data.size() > 0);
        std::cout << "10 000 ROW DATABASE" << std::endl;
        std::cout << getDuration(start, stop, Milliseconds) << std::endl;
    }

    SECTION("100 000 ROWS READ CSVBENCHMARK")
    {
        auto start = startBenchmark();
        auto data = readCSV<std::string>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
        auto stop = stopBenchmark();

        CHECK(data.size() > 0);
        std::cout << "100 000 ROW DATABASE" << std::endl;
        std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
    }
}