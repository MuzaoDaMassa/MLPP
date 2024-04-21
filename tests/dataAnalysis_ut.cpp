#include "catch.hpp"
#include "../src/mlpp.hpp"
#include "testUtils.hpp"

using namespace Utils;
using namespace Benchmark;
using namespace MLPP;

TEST_CASE("READ CSV METHODS")
{
    auto data1 = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
    auto data2 = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
    auto data3 = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");

    REQUIRE(data1.size() > 0);

    for (const auto& row : data1) {
        for (const auto& element : row) {
            CHECK(!element.empty());
        }
    }

    REQUIRE(data2.size() > 0);
 
    for (const auto& row : data2) {
        for (const auto& element : row) {
            CHECK(!element.empty());
        }
    }

    REQUIRE(data3.size() > 0);

    for (const auto& row : data3) {
        for (const auto& element : row) {
            CHECK(!element.empty());
        }
    }
}

TEST_CASE("MATRIX CONVERTER METHODS")
{
    auto data1 = DataAnalysis::DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
    auto data2 = DataAnalysis::DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
    auto data3 = DataAnalysis::DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");

    REQUIRE(data1.size() > 0);
    REQUIRE(data2.size() > 0);
    REQUIRE(data3.size() > 0);

    SECTION("100 ROWS DATASET CONVERSION TO FLOAT")
    {
        auto cData = DataAnalysis::matrix_converter<float>(data1);

        if (!cData.empty()) {
            for (const auto& row : cData) {
                for (const auto& element : row) {
                    REQUIRE(element >= 0);
                }
            }
        }
    }

    SECTION("100 ROWS DATASET CONVERSION TO DOUBLE")
    {
        auto cData = DataAnalysis::matrix_converter<double>(data1);

        if (!cData.empty()) {
            for (const auto& row : cData) {
                for (const auto& element : row) {
                    REQUIRE(element >= 0);
                }
            }
        }
    }

    SECTION("10 000 ROWS DATASET CONVERSION TO FLOAT")
    {
        auto cData = DataAnalysis::matrix_converter<float>(data2);

        if (!cData.empty()) {
            for (const auto& row : cData) {
                for (const auto& element : row) {
                    REQUIRE(element >= 0);
                }
            }
        }
    }

    SECTION("10 000  ROWS DATASET CONVERSION TO DOUBLE")
    {
        auto cData = DataAnalysis::matrix_converter<double>(data2);

        if (!cData.empty()) {
            for (const auto& row : cData) {
                for (const auto& element : row) {
                    REQUIRE(element >= 0);
                }
            }
        }
    }

    SECTION("100 000 ROWS DATASET CONVERSION TO FLOAT")
    {
        auto cData = DataAnalysis::matrix_converter<float>(data3);

        if (!cData.empty()) {
            for (const auto& row : cData) {
                for (const auto& element : row) {
                    REQUIRE(element >= 0);
                }
            }
        }
    }

    SECTION("100 000 ROWS DATASET CONVERSION TO DOUBLE")
    {
        auto cData = DataAnalysis::matrix_converter<double>(data3);

        if (!cData.empty()) {
            for (const auto& row : cData) {
                for (const auto& element : row) {
                    REQUIRE(element >= 0);
                }
            }
        }
    }
}

TEST_CASE("MATRIX FORMATTER METHODS")
{
    auto data1 = DataAnalysis::DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
    auto data2 = DataAnalysis::DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
    auto data3 = DataAnalysis::DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");

    REQUIRE(data1.size() > 0);
    REQUIRE(data2.size() > 0);
    REQUIRE(data3.size() > 0);

    SECTION("100 ROWS DATASET REMOVE SINGLE METHODS")
    {
        int rowToRemove = 50;
        int columnToRemove = 6;
        Mat2d<int> racToRemove = {{rowToRemove - 1}, {columnToRemove - 1}};
        auto oData = data1;

        // Remove single row
        DataAnalysis::matrix_formatter(data1, ROW, rowToRemove);
        auto fData1 = data1;

        REQUIRE(fData1.size() < oData.size());

        // Remove single column
        DataAnalysis::matrix_formatter(data1, COLUMN, columnToRemove);
        auto fData2 = data1;

        REQUIRE(fData2[0].size() < oData[0].size());

        // Remove single row and column
        DataAnalysis::matrix_formatter(data1, ROWANDCOLUMN, racToRemove);

        REQUIRE(data1.size() < fData1.size());
        REQUIRE(data1[0].size() < fData2[0].size());
    }

    SECTION("100 ROWS DATASET REMOVE MULTIPLE METHODS")
    {
        std::vector<int> rowsToRemove{10, 20, 30, 40, 50};
        std::vector<int> columnsToRemove{2, 4, 6};
        auto oData = data1;

        // Remove multiple rows
        DataAnalysis::matrix_formatter(data1, ROW, rowsToRemove);
        auto fData1 = data1;

        REQUIRE(fData1.size() < oData.size());

        // Remove multiple columns
        DataAnalysis::matrix_formatter(data1, COLUMN, columnsToRemove);
        auto fData2 = data1;

        REQUIRE(fData2[0].size() < oData[0].size());

        // Remove multiple rows and columns
        subtractAllElements(rowsToRemove, 1);
        subtractAllElements(columnsToRemove, 1);
        Mat2d<int> racToRemove = {{rowsToRemove}, {columnsToRemove}};
        DataAnalysis::matrix_formatter(data1, ROWANDCOLUMN, racToRemove);

        REQUIRE(data1.size() < fData1.size());
        REQUIRE(data1[0].size() < fData2[0].size());
    }

    SECTION("10 000 ROWS DATASET REMOVE SINGLE METHODS")
    {
        int rowToRemove = 5000;
        int columnToRemove = 6;
        Mat2d<int> racToRemove = {{rowToRemove - 1000}, {columnToRemove - 1}};
        auto oData = data2;

        // Remove single row
        DataAnalysis::matrix_formatter(data2, ROW, rowToRemove);
        auto fData1 = data2;

        REQUIRE(fData1.size() < oData.size());

        // Remove single column
        DataAnalysis::matrix_formatter(data2, COLUMN, columnToRemove);
        auto fData2 = data2;

        REQUIRE(fData2[0].size() < oData[0].size());

        // Remove single row and column
        DataAnalysis::matrix_formatter(data2, ROWANDCOLUMN, racToRemove);

        REQUIRE(data2.size() < fData1.size());
        REQUIRE(data2[0].size() < fData2[0].size());
    }

    SECTION("10 000 ROWS DATASET REMOVE MULTIPLE METHODS")
    {
        std::vector<int> rowsToRemove{1000, 2000, 3000, 4000, 5000};
        std::vector<int> columnsToRemove{2, 4, 6};
        auto oData = data2;

        // Remove multiple rows
        DataAnalysis::matrix_formatter(data2, ROW, rowsToRemove);
        auto fData1 = data2;

        REQUIRE(fData1.size() < oData.size());

        // Remove multiple columns
        DataAnalysis::matrix_formatter(data2, COLUMN, columnsToRemove);
        auto fData2 = data2;

        REQUIRE(fData2[0].size() < oData[0].size());

        // Remove multiple rows and columns
        subtractAllElements(rowsToRemove, 1);
        subtractAllElements(columnsToRemove, 1);
        Mat2d<int> racToRemove = {{rowsToRemove}, {columnsToRemove}};
        DataAnalysis::matrix_formatter(data2, ROWANDCOLUMN, racToRemove);

        REQUIRE(data2.size() < fData1.size());
        REQUIRE(data2[0].size() < fData2[0].size());
    }

    SECTION("100 000 ROWS DATASET REMOVE SINGLE METHODS")
    {
        int rowToRemove = 50000;
        int columnToRemove = 6;
        Mat2d<int> racToRemove = {{rowToRemove - 10000}, {columnToRemove - 1}};
        auto oData = data3;

        // Remove single row
        DataAnalysis::matrix_formatter(data3, ROW, rowToRemove);
        auto fData1 = data3;

        REQUIRE(fData1.size() < oData.size());

        // Remove single column
        DataAnalysis::matrix_formatter(data3, COLUMN, columnToRemove);
        auto fData2 = data3;

        REQUIRE(fData2[0].size() < oData[0].size());

        // Remove single row and column
        DataAnalysis::matrix_formatter(data3, ROWANDCOLUMN, racToRemove);

        REQUIRE(data3.size() < fData1.size());
        REQUIRE(data3[0].size() < fData2[0].size());
    }

    SECTION("100 000 ROWS DATASET REMOVE MULTIPLE METHODS")
    {
        std::vector<int> rowsToRemove{10000, 20000, 30000, 40000, 50000};
        std::vector<int> columnsToRemove{2, 4, 6};
        auto oData = data3;

        // Remove multiple rows
        DataAnalysis::matrix_formatter(data3, ROW, rowsToRemove);
        auto fData1 = data3;

        REQUIRE(fData1.size() < oData.size());

        // Remove multiple columns
        DataAnalysis::matrix_formatter(data3, COLUMN, columnsToRemove);
        auto fData2 = data3;
        
        REQUIRE(fData2[0].size() < oData[0].size());

        // Remove multiple rows and columns
        subtractAllElements(rowsToRemove, 1);
        subtractAllElements(columnsToRemove, 1);
        Mat2d<int> racToRemove = {{rowsToRemove}, {columnsToRemove}};
        DataAnalysis::matrix_formatter(data3, ROWANDCOLUMN, racToRemove);
        
        REQUIRE(data3.size() < fData1.size());
        REQUIRE(data3[0].size() < fData2[0].size());
    }
}

TEST_CASE("FIND METHODS")
{
    auto data1 = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
    auto data2 = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
    auto data3 = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");

    auto cData2 = DataAnalysis::matrix_converter<float>(data2);
    auto cData3 = DataAnalysis::matrix_converter<double>(data3);

    REQUIRE(!data1.empty());
    REQUIRE(!cData2.empty());
    REQUIRE(!cData3.empty());

    SECTION("FIND METHOD")
    {
        // In find method, only returns 0 in both values if nothing was found
        auto pos1 = DataAnalysis::find<std::string>(data1, "http://www.hatfield-saunders.net/");
        auto pos2 = DataAnalysis::find<float>(cData2, 2020.0F);
        auto pos3 = DataAnalysis::find<double>(cData3, 2021.0);

        REQUIRE(pos1[0] != 0);
        REQUIRE(pos1[1] != 0);
        REQUIRE(!pos2.empty());
        REQUIRE(!pos2.empty());
        REQUIRE(!pos3.empty());
        REQUIRE(!pos3.empty());
    }

    SECTION("FIND BY POSITION METHOD")
    {
        // Create vector to hold positions to search
        std::vector<int> pos1{44, 4};
        std::vector<int> pos2{5555, 5};
        std::vector<int> pos3{66666, 6};

        // Return element from method
        auto el1 = DataAnalysis::find_by_pos<std::string>(data1, pos1);
        auto el2 = DataAnalysis::find_by_pos<float>(cData2, pos2);
        auto el3 = DataAnalysis::find_by_pos<double>(cData3, pos3);

        REQUIRE(!el1.empty());
        REQUIRE(el2 >= 0);
        REQUIRE(el3 >= 0);
    }

    SECTION("FIND ALL METHODS")
    {
        // In find method, matrix returns empty if nothing was found
        auto pos1 = DataAnalysis::findAll<std::string>(data1, "Netherlands");
        auto pos2 = DataAnalysis::findAll<float>(cData2, 2020.0F);
        auto pos3 = DataAnalysis::findAll<double>(cData3, 2021.0);

        std::cout << "100 ROW DATABASE" << std::endl;
        REQUIRE(!pos1.empty());
        std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos1.size() << std::endl;

        std::cout << "10 000 ROW DATABASE" << std::endl;
        REQUIRE(!pos2.empty());
        std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos2.size() << std::endl;

        std::cout << "100 000 ROW DATABASE" << std::endl;
        REQUIRE(!pos3.empty());
        std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos3.size() << std::endl;
    }
}

TEST_CASE("READ METHODS BENCHMARKS", "[.benchmarks]")
{
    SECTION("100 ROWS READ CSV BENCHMARK")
    {
        auto start = startBenchmark();
        auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
        auto stop = stopBenchmark();

        std::cout << "100 ROW DATABASE READ CSV - STRING" << std::endl;
        std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
    }

    SECTION("10 000 ROWS READ CSV BENCHMARK - STRING")
    {
        auto start = startBenchmark();
        auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
        auto stop = stopBenchmark();

        std::cout << "10 000 ROW DATABASE READ CSV" << std::endl;
        std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
    }

    SECTION("100 000 ROWS READ CSV BENCHMARK ")
    {
        auto start = startBenchmark();
        auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
        auto stop = stopBenchmark();

        std::cout << "100 000 ROW DATABASE READ CSV" << std::endl;
        std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
    }
}

TEST_CASE("MATRIX CONVERSION METHODS BENCHMARKS", "[.benchmarks]")
{
    auto data1 = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
    auto data2 = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
    auto data3 = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");

    SECTION("100 ROWS DATASET CONVERSION TO FLOAT")
    {
        auto start = startBenchmark();
        auto cData = DataAnalysis::matrix_converter<float>(data1);
        auto stop = stopBenchmark();

        if (!cData.empty())
        {
            std::cout << "100 ROW DATABASE CONVERTO TO FLOAT BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }
    }

    SECTION("100 ROWS DATASET CONVERSION TO DOUBLE")
    {
        auto start = startBenchmark();
        auto cData = DataAnalysis::matrix_converter<double>(data1);
        auto stop = stopBenchmark();

        if (!cData.empty())
        {
            std::cout << "100 ROW DATABASE CONVERTO TO DOUBLE BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }
    }

    SECTION("10 000 ROWS DATASET CONVERSION TO FLOAT")
    {
        auto start = startBenchmark();
        auto cData = DataAnalysis::matrix_converter<float>(data2);
        auto stop = stopBenchmark();

        if (!cData.empty())
        {
            std::cout << "10 000 ROW DATABASE CONVERTO TO FLOAT BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }
    }

    SECTION("10 000  ROWS DATASET CONVERSION TO DOUBLE")
    {
        auto start = startBenchmark();
        auto cData = DataAnalysis::matrix_converter<double>(data2);
        auto stop = stopBenchmark();

        if (!cData.empty())
        {
            std::cout << "10 000 ROW DATABASE CONVERTO TO DOUBLE BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }
    }

    SECTION("100 000 ROWS DATASET CONVERSION TO FLOAT")
    {
        auto start = startBenchmark();
        auto cData = DataAnalysis::matrix_converter<float>(data3);
        auto stop = stopBenchmark();

        if (!cData.empty())
        {
            std::cout << "100 000 ROW DATABASE CONVERTO TO FLOAT BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }
    }

    SECTION("100 000 ROWS DATASET CONVERSION TO DOUBLE")
    {
        auto start = startBenchmark();
        auto cData = DataAnalysis::matrix_converter<double>(data3);
        auto stop = stopBenchmark();

        if (!cData.empty())
        {
            std::cout << "100 000 ROW DATABASE CONVERTO TO DOUBLE BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }
    }
}

TEST_CASE("MATRIX FORMATTER METHODS BENCHMARKS", "[.benchmarks]")
{
    auto data1 = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
    auto data2 = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
    auto data3 = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");

    SECTION("100 ROWS DATASET REMOVE SINGLE METHODS BENCHMARKS")
    {
        int rowToRemove = 50;
        int columnToRemove = 6;
        Mat2d<int> racToRemove = {{rowToRemove - 1}, {columnToRemove - 1}};

        SECTION("REMOVE SINGLE ROW BENCHMARK")
        {
            auto start = startBenchmark();
            DataAnalysis::matrix_formatter(data1, ROW, rowToRemove);
            auto stop = stopBenchmark();

            std::cout << "100 ROW REMOVE SINGLE ROW BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("REMOVE SINGLE COLUMN BENCHMARK")
        {
            auto start = startBenchmark();
            DataAnalysis::matrix_formatter(data1, COLUMN, columnToRemove);
            auto stop = stopBenchmark();

            std::cout << "100 ROW REMOVE SINGLE COLUMN BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("REMOVE SINGLE ROW AND COLUMN BENCHMARK")
        {
            auto start = startBenchmark();
            DataAnalysis::matrix_formatter(data1, ROWANDCOLUMN, racToRemove);
            auto stop = stopBenchmark();

            std::cout << "100 ROW REMOVE SINGLE ROW AND COLUMN BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }
    }

    SECTION("100 ROWS DATASET REMOVE MULTIPLE METHODS BENCHMARKS")
    {
        std::vector<int> rowsToRemove{10, 20, 30, 40, 50};
        std::vector<int> columnsToRemove{2, 4, 6};

        SECTION("REMOVE MULTIPLE ROWS BENCHMARK")
        {
            auto start = startBenchmark();
            DataAnalysis::matrix_formatter(data1, ROW, rowsToRemove);
            auto stop = stopBenchmark();

            std::cout << "100 ROW REMOVE MULTIPLE ROWS BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("REMOVE MULTIPLE COLUMNS BENCHMARK")
        {
            auto start = startBenchmark();
            DataAnalysis::matrix_formatter(data1, COLUMN, columnsToRemove);
            auto stop = stopBenchmark();

            std::cout << "100 ROW REMOVE MULTIPLE COLUMNS BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("REMOVE MULTIPLE ROWS AND COLUMNS BENCHMARK")
        {
            subtractAllElements(rowsToRemove, 1);
            subtractAllElements(columnsToRemove, 1);
            Mat2d<int> racToRemove = {{rowsToRemove}, {columnsToRemove}};
            auto start = startBenchmark();
            DataAnalysis::matrix_formatter(data1, ROWANDCOLUMN, racToRemove);
            auto stop = stopBenchmark();

            std::cout << "100 ROW REMOVE MULTIPLE ROWS AND COLUMNS BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }
    }

    SECTION("10 000 ROWS DATASET REMOVE SINGLE METHODS BENCHMARKS")
    {
        int rowToRemove = 5000;
        int columnToRemove = 6;
        Mat2d<int> racToRemove = {{rowToRemove - 1000}, {columnToRemove - 1}};

        SECTION("REMOVE SINGLE ROW BENCHMARK")
        {
            auto start = startBenchmark();
            DataAnalysis::matrix_formatter(data2, ROW, rowToRemove);
            auto stop = stopBenchmark();

            std::cout << "10 000 ROW REMOVE SINGLE ROW BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("REMOVE SINGLE COLUMN BENCHMARK")
        {
            auto start = startBenchmark();
            DataAnalysis::matrix_formatter(data2, COLUMN, columnToRemove);
            auto stop = stopBenchmark();

            std::cout << "10 000 ROW REMOVE SINGLE COLUMN BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("REMOVE SINGLE ROW AND COLUMN BENCHMARK")
        {
            auto start = startBenchmark();
            DataAnalysis::matrix_formatter(data2, ROWANDCOLUMN, racToRemove);
            auto stop = stopBenchmark();

            std::cout << "10 000 ROW REMOVE SINGLE ROW AND COLUMN BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }
    }

    SECTION("10 000 ROWS DATASET REMOVE MULTIPLE METHODS BENCHMARKS")
    {
        std::vector<int> rowsToRemove{1000, 2000, 3000, 4000, 5000};
        std::vector<int> columnsToRemove{2, 4, 6};

        SECTION("REMOVE MULTIPLE ROWS BENCHMARK")
        {
            auto start = startBenchmark();
            DataAnalysis::matrix_formatter(data2, ROW, rowsToRemove);
            auto stop = stopBenchmark();

            std::cout << "10 000 ROW REMOVE MULTIPLE ROWS BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("REMOVE MULTIPLE COLUMNS BENCHMARK")
        {
            auto start = startBenchmark();
            DataAnalysis::matrix_formatter(data2, COLUMN, columnsToRemove);
            auto stop = stopBenchmark();

            std::cout << "10 000 ROW REMOVE MULTIPLE COLUMNS BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("REMOVE MULTIPLE ROWS AND COLUMNS BENCHMARK")
        {
            subtractAllElements(rowsToRemove, 1);
            subtractAllElements(columnsToRemove, 1);
            Mat2d<int> racToRemove = {{rowsToRemove}, {columnsToRemove}};
            auto start = startBenchmark();
            DataAnalysis::matrix_formatter(data2, ROWANDCOLUMN, racToRemove);
            auto stop = stopBenchmark();

            std::cout << "10 000 ROW REMOVE MULTIPLE ROWS AND COLUMNS BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }
    }

    SECTION("100 000 ROWS DATASET REMOVE SINGLE METHODS BENCHMARKS")
    {
        int rowToRemove = 50000;
        int columnToRemove = 6;
        Mat2d<int> racToRemove = {{rowToRemove - 10000}, {columnToRemove - 1}};

        SECTION("REMOVE SINGLE ROW BENCHMARK")
        {
            auto start = startBenchmark();
            DataAnalysis::matrix_formatter(data3, ROW, rowToRemove);
            auto stop = stopBenchmark();

            std::cout << "100 000 ROW REMOVE SINGLE ROW BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("REMOVE SINGLE COLUMN BENCHMARK")
        {
            auto start = startBenchmark();
            DataAnalysis::matrix_formatter(data3, COLUMN, columnToRemove);
            auto stop = stopBenchmark();

            std::cout << "100 000 ROW REMOVE SINGLE COLUMN BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("REMOVE SINGLE ROW AND COLUMN BENCHMARK")
        {
            auto start = startBenchmark();
            DataAnalysis::matrix_formatter(data3, ROWANDCOLUMN, racToRemove);
            auto stop = stopBenchmark();

            std::cout << "100 000 ROW REMOVE SINGLE ROW AND COLUMN BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }
    }

    SECTION("100 000 ROWS DATASET REMOVE MULTIPLE METHODS BENCHMARKS")
    {
        std::vector<int> rowsToRemove{10000, 20000, 30000, 40000, 50000};
        std::vector<int> columnsToRemove{2, 4, 6};

        SECTION("REMOVE MULTIPLE ROWS BENCHMARK")
        {
            auto start = startBenchmark();
            DataAnalysis::matrix_formatter(data3, ROW, rowsToRemove);
            auto stop = stopBenchmark();

            std::cout << "100 000 ROW REMOVE MULTIPLE ROWS BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("REMOVE MULTIPLE COLUMNS BENCHMARK")
        {
            auto start = startBenchmark();
            DataAnalysis::matrix_formatter(data3, COLUMN, columnsToRemove);
            auto stop = stopBenchmark();

            std::cout << "100 000 ROW REMOVE MULTIPLE COLUMNS BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("REMOVE MULTIPLE ROWS AND COLUMNS BENCHMARK")
        {
            subtractAllElements(rowsToRemove, 1);
            subtractAllElements(columnsToRemove, 1);
            Mat2d<int> racToRemove = {{rowsToRemove}, {columnsToRemove}};
            auto start = startBenchmark();
            DataAnalysis::matrix_formatter(data3, ROWANDCOLUMN, racToRemove);
            auto stop = stopBenchmark();

            std::cout << "100 000 ROW REMOVE MULTIPLE ROWS AND COLUMNS BENCHMARK" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }
    }
}

TEST_CASE("FIND METHODS BENCHMARKS", "[.benchmarks]")
{
    SECTION("FIND METHOD BENCHMARKS")
    {
        SECTION("100 ROWS FIND METHOD BENCHMARK - STRING")
        {
            // For testing purposes, the desired element will always be the last possible
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");

            auto start = startBenchmark();
            auto pos = DataAnalysis::find<std::string>(data, "http://www.hatfield-saunders.net/");
            auto stop = stopBenchmark();

            std::cout << "100 ROW DATABASE FIND" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 ROWS FIND METHOD BENCHMARK - FLOAT")
        {
            // For testing purposes, the desired element will always be the last possible
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
            auto cData = DataAnalysis::matrix_converter<float>(data);

            auto start = startBenchmark();
            auto pos = DataAnalysis::find<float>(cData, 783.639F);
            auto stop = stopBenchmark();

            std::cout << "100 ROW DATABASE FIND - FLOAT" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 ROWS FIND METHOD BENCHMARK - DOUBLE")
        {
            // For testing purposes, the desired element will always be the last possible
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
            auto cData = DataAnalysis::matrix_converter<double>(data);

            auto start = startBenchmark();
            auto pos = DataAnalysis::find<double>(cData, 783.639);
            auto stop = stopBenchmark();

            std::cout << "100 ROW DATABASE FIND - DOUBLE" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("10 000 ROWS FIND METHOD BENCHMARK - STRING")
        {
            // For testing purposes, the desired element will always be the last possible
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");

            auto start = startBenchmark();
            auto pos = DataAnalysis::find<std::string>(data, "http://nixon.net/");
            auto stop = stopBenchmark();

            std::cout << "10 000 ROW DATABASE FIND - STRING" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("10 000 ROWS FIND METHOD BENCHMARK - FLOAT")
        {
            // For testing purposes, the desired element will always be the last possible
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
            auto cData = DataAnalysis::matrix_converter<float>(data);

            auto start = startBenchmark();
            auto pos = DataAnalysis::find<float>(cData, 423.632F);
            auto stop = stopBenchmark();

            std::cout << "10 000 ROW DATABASE FIND - FLOAT" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("10 000 ROWS FIND METHOD BENCHMARK - DOUBLE")
        {
            // For testing purposes, the desired element will always be the last possible
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
            auto cData = DataAnalysis::matrix_converter<double>(data);

            auto start = startBenchmark();
            auto pos = DataAnalysis::find<double>(cData, 423.632);
            auto stop = stopBenchmark();

            std::cout << "10 000 ROW DATABASE FIND - DOUBLE" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 000 ROWS FIND METHOD BENCHMARK - STRING")
        {
            // For testing purposes, the desired element will always be the last possible
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");

            auto start = startBenchmark();
            auto pos = DataAnalysis::find<std::string>(data, "https://www.walter.com/");
            auto stop = stopBenchmark();

            std::cout << "100 000 ROW DATABASE FIND - STRING" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 000 ROWS FIND METHOD BENCHMARK - FLOAT")
        {
            // For testing purposes, the desired element will always be the last possible
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
            auto cData = DataAnalysis::matrix_converter<float>(data);

            auto start = startBenchmark();
            auto pos = DataAnalysis::find<float>(cData, 157.189F);
            auto stop = stopBenchmark();

            std::cout << "100 000 ROW DATABASE FIND - FLOAT" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 000 ROWS FIND METHOD BENCHMARK - DOUBLE")
        {
            // For testing purposes, the desired element will always be the last possible
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
            auto cData = DataAnalysis::matrix_converter<double>(data);

            auto start = startBenchmark();
            auto pos = DataAnalysis::find<double>(cData, 157.189);
            auto stop = stopBenchmark();

            std::cout << "100 000 ROW DATABASE FIND - DOUBLE" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }
    }

    SECTION("FIND BY POSITION METHOD BENCHMARKS")
    {
        SECTION("100 ROWS FIND BY POSITION METHOD - STRING")
        {
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
            std::vector<int> pos{44, 4};

            auto start = startBenchmark();
            auto el = DataAnalysis::find_by_pos<std::string>(data, pos);
            auto stop = stopBenchmark();

            std::cout << "100 ROW DATABASE FIND BY POSITION - STRING" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 ROWS FIND BY POSITION METHOD - FLOAT")
        {
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
            auto cData = DataAnalysis::matrix_converter<float>(data);
            std::vector<int> pos{55, 5};

            auto start = startBenchmark();
            auto el = DataAnalysis::find_by_pos<float>(cData, pos);
            auto stop = stopBenchmark();

            std::cout << "100 ROW DATABASE FIND BY POSITION - FLOAT" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 ROWS FIND BY POSITION METHOD - DOUBLE")
        {
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
            auto cData = DataAnalysis::matrix_converter<double>(data);
            std::vector<int> pos{66, 6};

            auto start = startBenchmark();
            auto el = DataAnalysis::find_by_pos<double>(cData, pos);
            auto stop = stopBenchmark();

            std::cout << "100 ROW DATABASE FIND BY POSITION - DOUBLE" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("10 000 ROWS FIND BY POSITION METHOD - STRING")
        {
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
            std::vector<int> pos{4444, 4};

            auto start = startBenchmark();
            auto el = DataAnalysis::find_by_pos<std::string>(data, pos);
            auto stop = stopBenchmark();

            std::cout << "10 000 ROW DATABASE FIND BY POSITION - STRING" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("10 000 ROWS FIND BY POSITION METHOD - FLOAT")
        {
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
            auto cData = DataAnalysis::matrix_converter<float>(data);
            std::vector<int> pos{5555, 5};

            auto start = startBenchmark();
            auto el = DataAnalysis::find_by_pos<float>(cData, pos);
            auto stop = stopBenchmark();

            std::cout << "10 000 ROW DATABASE FIND BY POSITION - FLOAT" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("10 000 ROWS FIND BY POSITION METHOD - DOUBLE")
        {
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
            auto cData = DataAnalysis::matrix_converter<double>(data);
            std::vector<int> pos{6666, 6};

            auto start = startBenchmark();
            auto el = DataAnalysis::find_by_pos<double>(cData, pos);
            auto stop = stopBenchmark();

            std::cout << "10 000 ROW DATABASE FIND BY POSITION - DOUBLE" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 000 ROWS FIND BY POSITION METHOD - STRING")
        {
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
            std::vector<int> pos{44444, 4};

            auto start = startBenchmark();
            auto el = DataAnalysis::find_by_pos<std::string>(data, pos);
            auto stop = stopBenchmark();

            std::cout << "100 000 ROW DATABASE FIND BY POSITION - STRING" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 000 ROWS FIND BY POSITION METHOD - FLOAT")
        {
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
            auto cData = DataAnalysis::matrix_converter<float>(data);
            std::vector<int> pos{55555, 5};

            auto start = startBenchmark();
            auto el = DataAnalysis::find_by_pos<float>(cData, pos);
            auto stop = stopBenchmark();

            std::cout << "100 000 ROW DATABASE FIND BY POSITION - FLOAT" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 000 ROWS FIND BY POSITION METHOD - DOUBLE")
        {
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
            auto cData = DataAnalysis::matrix_converter<double>(data);
            std::vector<int> pos{66666, 6};

            auto start = startBenchmark();
            auto el = DataAnalysis::find_by_pos<double>(cData, pos);
            auto stop = stopBenchmark();

            std::cout << "100 000 ROW DATABASE FIND BY POSITION - DOUBLE" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }
    }

    SECTION("FIND ALL METHOD BENCHMARKS")
    {
        SECTION("100 ROWS FIND ALL METHOD BENCHMARK - STRING")
        {
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");

            auto start = startBenchmark();
            auto pos = DataAnalysis::findAll<std::string>(data, "Netherlands");
            auto stop = stopBenchmark();

            std::cout << "100 ROW DATABASE FIND ALL - STRING" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 ROWS FIND ALL METHOD BENCHMARK - FLOAT")
        {
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
            auto cData = DataAnalysis::matrix_converter<float>(data);

            auto start = startBenchmark();
            auto pos = DataAnalysis::findAll<float>(cData, 2020.0F);
            auto stop = stopBenchmark();

            std::cout << "100 ROW DATABASE FIND ALL - FLOAT" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 ROWS FIND ALL METHOD BENCHMARK - DOUBLE")
        {
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
            auto cData = DataAnalysis::matrix_converter<double>(data);

            auto start = startBenchmark();
            auto pos = DataAnalysis::findAll<double>(cData, 2021.0);
            auto stop = stopBenchmark();

            std::cout << "100 ROW DATABASE FIND ALL - DOUBLE" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("10 000 ROWS FIND ALL METHOD BENCHMARK - STRING")
        {
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");

            auto start = startBenchmark();
            auto pos = DataAnalysis::findAll<std::string>(data, "Netherlands");
            auto stop = stopBenchmark();

            std::cout << "10 000 ROW DATABASE FIND ALL - STRING" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("10 000 ROWS FIND ALL METHOD BENCHMARK - FLOAT")
        {
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
            auto cData = DataAnalysis::matrix_converter<float>(data);

            auto start = startBenchmark();
            auto pos = DataAnalysis::findAll<float>(cData, 2020.0F);
            auto stop = stopBenchmark();

            std::cout << "10 000 ROW DATABASE FIND ALL - FLOAT" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("10 000 ROWS FIND ALL METHOD BENCHMARK - DOUBLE")
        {
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
            auto cData = DataAnalysis::matrix_converter<double>(data);

            auto start = startBenchmark();
            auto pos = DataAnalysis::findAll<double>(cData, 2021.0);
            auto stop = stopBenchmark();

            std::cout << "10 000 ROW DATABASE FIND ALL - DOUBLE" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 000 ROWS FIND ALL METHOD BENCHMARK - STRING")
        {
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");

            auto start = startBenchmark();
            auto pos = DataAnalysis::findAll<std::string>(data, "Netherlands");
            auto stop = stopBenchmark();

            std::cout << "100 000 ROW DATABASE FIND ALL - STRING" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 000 ROWS FIND ALL METHOD BENCHMARK - FLOAT")
        {
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
            auto cData = DataAnalysis::matrix_converter<float>(data);

            auto start = startBenchmark();
            auto pos = DataAnalysis::findAll<float>(cData, 2020.0F);
            auto stop = stopBenchmark();

            std::cout << "100 000 ROW DATABASE FIND ALL - FLOAT" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 000 ROWS FIND ALL METHOD BENCHMARK - DOUBLE")
        {
            auto data = DataAnalysis::read_csv_file("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
            auto cData = DataAnalysis::matrix_converter<double>(data);

            auto start = startBenchmark();
            auto pos = DataAnalysis::findAll<double>(cData, 2021.0);
            auto stop = stopBenchmark();

            std::cout << "100 000 ROW DATABASE FIND ALL - DOUBLE" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }
    }
}