#include "catch.hpp"
#include "../src/mlpp.hpp"
#include "testUtils.hpp"

using namespace Benchmark;
using namespace DataAnalysis;

TEST_CASE ("Data Analysis Unit Testing")
{
    SECTION ("READ CSV METHODS")
    {
        auto data1 = readCSV<std::string>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
        auto data2 = readCSV<float>("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
        auto data3 = readCSV<double>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv"); 

        REQUIRE(data1.size() > 0); 

        for (const auto& row : data1) 
        {
            for (const auto& element : row) 
            {
                CHECK(!element.empty());
            }
        } 

        REQUIRE(data2.size() > 0); 

        for (const auto& row : data2) 
        {
            for (const auto& element : row) 
            {
                CHECK(element >= 0);
            }
        } 

        REQUIRE(data3.size() > 0);   

        for (const auto& row : data3) 
        {
            for (const auto& element : row) 
            {
                CHECK(element >= 0);
            }
        } 
    }

    SECTION("FIND METHODS")
    {
        auto data1 = readCSV<std::string>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
        auto data2 = readCSV<float>("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
        auto data3 = readCSV<double>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv"); 

        REQUIRE(data1.size() > 0); 
        REQUIRE(data2.size() > 0); 
        REQUIRE(data3.size() > 0); 

        SECTION("FIND METHOD")
        {
            // In find method, only returns 0 in both values if nothing was found 
            auto pos1 = find<std::string>(data1, "tuckerangie@salazar.net");
            auto pos2 = find<float>(data2, 2020.0);
            auto pos3 = find<double>(data3, 2021.0);

            REQUIRE(pos1[0] != 0);
            REQUIRE(pos1[1] != 0);
            REQUIRE(pos2[0] != 0);
            REQUIRE(pos2[1] != 0);
            REQUIRE(pos3[0] != 0);
            REQUIRE(pos3[1] != 0);
        }

        SECTION("FIND BY POSITION METHOD")
        {
            
        }

        SECTION("FIND ALL METHODS")
        {
            // In find method, matrix returns empty if nothing was found 
            auto pos1 = findAll<std::string>(data1, "Netherlands");
            auto pos2 = findAll<float>(data2, 2020.0F);
            auto pos3 = findAll<double>(data3, 2021.0);

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
}

TEST_CASE("Data Analysis Unit Benchmarking", "[.benchmark]")
{
    SECTION("ASSORTED READ METHODS BENCHMARKS")
    {
        SECTION("100 ROWS READ CSV BENCHMARK - STRING")
        {
            auto start = startBenchmark();
            auto data = readCSV<std::string>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
            auto stop = stopBenchmark();

            REQUIRE(data.size() > 0);
            std::cout << "100 ROW DATABASE READ CSV - STRING" << std::endl;
            std::cout << getDuration(start, stop, Microseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 ROWS READ CSV BENCHMARK - FLOAT")
        {
            auto start = startBenchmark();
            auto data = readCSV<float>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
            auto stop = stopBenchmark();

            REQUIRE(data.size() > 0);
            std::cout << "100 ROW DATABASE READ CSV - FLOAT" << std::endl;
            std::cout << getDuration(start, stop, Microseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 ROWS READ CSV BENCHMARK - DOUBLE")
        {
            auto start = startBenchmark();
            auto data = readCSV<double>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
            auto stop = stopBenchmark();

            REQUIRE(data.size() > 0);
            std::cout << "100 ROW DATABASE READ CSV - DOUBLE" << std::endl;
            std::cout << getDuration(start, stop, Microseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("10 000 ROWS READ CSV BENCHMARK - STRING")
        {
            auto start = startBenchmark();
            auto data = readCSV<std::string>("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
            auto stop = stopBenchmark();

            REQUIRE(data.size() > 0);
            std::cout << "10 000 ROW DATABASE READ CSV - STRING" << std::endl;
            std::cout << getDuration(start, stop, Microseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("10 000 ROWS READ CSV BENCHMARK - FLOAT")
        {
            auto start = startBenchmark();
            auto data = readCSV<float>("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
            auto stop = stopBenchmark();

            REQUIRE(data.size() > 0);
            std::cout << "10 000 ROW DATABASE READ CSV - FLOAT" << std::endl;
            std::cout << getDuration(start, stop, Microseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("10 000 ROWS READ CSV BENCHMARK - DOUBLE")
        {
            auto start = startBenchmark();
            auto data = readCSV<double>("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
            auto stop = stopBenchmark();

            REQUIRE(data.size() > 0);
            std::cout << "10 000 ROW DATABASE READ CSV - DOUBLE" << std::endl;
            std::cout << getDuration(start, stop, Microseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 000 ROWS READ CSV BENCHMARK - STRING")
        {
            auto start = startBenchmark();
            auto data = readCSV<std::string>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
            auto stop = stopBenchmark();

            REQUIRE(data.size() > 0);
            std::cout << "100 000 ROW DATABASE READ CSV - STRING" << std::endl;
            std::cout << getDuration(start, stop, Microseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 000 ROWS READ CSVBENCHMARK - FLOAT")
        {
            auto start = startBenchmark();
            auto data = readCSV<float>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
            auto stop = stopBenchmark();

            REQUIRE(data.size() > 0);
            std::cout << "100 000 ROW DATABASE READ CSV - FLOAT" << std::endl;
            std::cout << getDuration(start, stop, Microseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 000 ROWS READ CSVBENCHMARK - DOUBLE")
        {
            auto start = startBenchmark();
            auto data = readCSV<double>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
            auto stop = stopBenchmark();

            REQUIRE(data.size() > 0);
            std::cout << "100 000 ROW DATABASE READ CSV - DOUBLE" << std::endl;
            std::cout << getDuration(start, stop, Microseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

    }

    SECTION("ASSORTED FIND METHODS BENCHMARKS")
    {
        SECTION("FIND METHOD BENCHMARKS")
        {
            SECTION("100 ROWS FIND METHOD BENCHMARK - STRING")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV<std::string>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = find<std::string>(data, "http://www.hatfield-saunders.net/");
                auto stop = stopBenchmark();

                CHECK(pos[0] != 0);
                CHECK(pos[1] != 0);

                std::cout << "100 ROW DATABASE FIND - STRING" << std::endl;
                std::cout << getDuration(start, stop, Microseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 ROWS FIND METHOD BENCHMARK - FLOAT")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV<float>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = find<float>(data, 783.639F);
                auto stop = stopBenchmark();

                CHECK(pos[0] != 0);
                CHECK(pos[1] != 0);

                std::cout << "100 ROW DATABASE FIND - FLOAT" << std::endl;
                std::cout << getDuration(start, stop, Microseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 ROWS FIND METHOD BENCHMARK - DOUBLE")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV<double>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = find<double>(data, 783.639);
                auto stop = stopBenchmark();

                CHECK(pos[0] != 0);
                CHECK(pos[1] != 0);

                std::cout << "100 ROW DATABASE FIND - DOUBLE" << std::endl;
                std::cout << getDuration(start, stop, Microseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("10 000 ROWS FIND METHOD BENCHMARK - STRING")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV<std::string>("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = find<std::string>(data, "http://nixon.net/");
                auto stop = stopBenchmark();

                CHECK(pos[0] != 0);
                CHECK(pos[1] != 0);

                std::cout << "10 000 ROW DATABASE FIND - STRING" << std::endl;
                std::cout << getDuration(start, stop, Microseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("10 000 ROWS FIND METHOD BENCHMARK - FLOAT")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV<float>("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = find<float>(data, 423.632F);
                auto stop = stopBenchmark();

                CHECK(pos[0] != 0);
                CHECK(pos[1] != 0);

                std::cout << "10 000 ROW DATABASE FIND - FLOAT" << std::endl;
                std::cout << getDuration(start, stop, Microseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("10 000 ROWS FIND METHOD BENCHMARK - DOUBLE")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV<double>("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = find<double>(data, 423.632);
                auto stop = stopBenchmark();

                CHECK(pos[0] != 0);
                CHECK(pos[1] != 0);

                std::cout << "10 000 ROW DATABASE FIND - DOUBLE" << std::endl;
                std::cout << getDuration(start, stop, Microseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 000 ROWS FIND METHOD BENCHMARK - STRING")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV<std::string>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = find<std::string>(data, "https://www.walter.com/");
                auto stop = stopBenchmark();

                CHECK(pos[0] != 0);
                CHECK(pos[1] != 0);

                std::cout << "100 000 ROW DATABASE FIND - STRING" << std::endl;
                std::cout << getDuration(start, stop, Microseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 000 ROWS FIND METHOD BENCHMARK - FLOAT")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV<float>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = find<float>(data, 157.189F);
                auto stop = stopBenchmark();

                CHECK(pos[0] != 0);
                CHECK(pos[1] != 0);

                std::cout << "100 000 ROW DATABASE FIND - FLOAT" << std::endl;
                std::cout << getDuration(start, stop, Microseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 000 ROWS FIND METHOD BENCHMARK - DOUBLE")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV<double>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = find<double>(data, 157.189);
                auto stop = stopBenchmark();

                CHECK(pos[0] != 0);
                CHECK(pos[1] != 0);

                std::cout << "100 000 ROW DATABASE FIND - DOUBLE" << std::endl;
                std::cout << getDuration(start, stop, Microseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }
           
        }

        SECTION("FIND ALL METHOD BENCHMARKS")
        {
            SECTION("100 ROWS FIND ALL METHOD BENCHMARK - STRING")
            {
                auto data = readCSV<std::string>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = findAll<std::string>(data, "Netherlands");
                auto stop = stopBenchmark();

                CHECK(!pos.empty());

                std::cout << "100 ROW DATABASE FIND ALL - STRING" << std::endl;
                std::cout << getDuration(start, stop, Microseconds) << std::endl;
                std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 ROWS FIND ALL METHOD BENCHMARK - FLOAT")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV<float>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = findAll<float>(data, 2020.0F);
                auto stop = stopBenchmark();

                CHECK(!pos.empty());

                std::cout << "100 ROW DATABASE FIND ALL - FLOAT" << std::endl;
                std::cout << getDuration(start, stop, Microseconds) << std::endl;
                std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 ROWS FIND ALL METHOD BENCHMARK - DOUBLE")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV<double>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = findAll<double>(data, 2021.0);
                auto stop = stopBenchmark();

                CHECK(!pos.empty());

                std::cout << "100 ROW DATABASE FIND ALL - DOUBLE" << std::endl;
                std::cout << getDuration(start, stop, Microseconds) << std::endl;
                std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("10 000 ROWS FIND ALL METHOD BENCHMARK - STRING")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV<std::string>("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = findAll<std::string>(data, "Netherlands");
                auto stop = stopBenchmark();

                CHECK(!pos.empty());

                std::cout << "10 000 ROW DATABASE FIND ALL - STRING" << std::endl;
                std::cout << getDuration(start, stop, Microseconds) << std::endl;
                std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("10 000 ROWS FIND ALL METHOD BENCHMARK - FLOAT")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV<float>("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = findAll<float>(data, 2020.0F);
                auto stop = stopBenchmark();

                CHECK(!pos.empty());

                std::cout << "10 000 ROW DATABASE FIND ALL - FLOAT" << std::endl;
                std::cout << getDuration(start, stop, Microseconds) << std::endl;
                std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("10 000 ROWS FIND ALL METHOD BENCHMARK - DOUBLE")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV<double>("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = findAll<double>(data, 2021.0);
                auto stop = stopBenchmark();

                CHECK(!pos.empty());

                std::cout << "10 000 ROW DATABASE FIND ALL - DOUBLE" << std::endl;
                std::cout << getDuration(start, stop, Microseconds) << std::endl;
                std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 000 ROWS FIND ALL METHOD BENCHMARK - STRING")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV<std::string>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = findAll<std::string>(data, "Netherlands");
                auto stop = stopBenchmark();

                CHECK(!pos.empty());

                std::cout << "100 000 ROW DATABASE FIND ALL - STRING" << std::endl;
                std::cout << getDuration(start, stop, Microseconds) << std::endl;
                std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 000 ROWS FIND ALL METHOD BENCHMARK - FLOAT")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV<float>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = findAll<float>(data, 2020.0F);
                auto stop = stopBenchmark();

                CHECK(!pos.empty());

                std::cout << "100 000 ROW DATABASE FIND ALL - FLOAT" << std::endl;
                std::cout << getDuration(start, stop, Microseconds) << std::endl;
                std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 000 ROWS FIND ALL METHOD BENCHMARK - DOUBLE")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV<double>("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = findAll<double>(data, 2021.0);
                auto stop = stopBenchmark();

                CHECK(!pos.empty());

                std::cout << "100 000 ROW DATABASE FIND ALL - DOUBLE" << std::endl;
                std::cout << getDuration(start, stop, Microseconds) << std::endl;
                std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }
        }
    }
}
        