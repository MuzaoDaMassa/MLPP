#include "catch.hpp"
#include "../src/mlpp.hpp"
#include "testUtils.hpp"

using namespace Benchmark;
using namespace DataAnalysis;

TEST_CASE ("Data Analysis Unit Testing")
{
    SECTION ("READ CSV METHODS")
    {
        auto data1 = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
        auto data2 = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
        auto data3 = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv"); 

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
                CHECK(!element.empty());
            }
        } 

        REQUIRE(data3.size() > 0);   

        for (const auto& row : data3) 
        {
            for (const auto& element : row) 
            {
                CHECK(!element.empty());
            }
        } 
    }

    SECTION("MATRIX CONVERTER METHODS")
    {
        auto data1 = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
        auto data2 = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
        auto data3 = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv"); 

        REQUIRE(data1.size() > 0); 
        REQUIRE(data2.size() > 0); 
        REQUIRE(data3.size() > 0); 

        SECTION("100 ROWS DATASET CONVERSION TO FLOAT")
        {
            auto cData = matrixConverter<float>(data1);
            
            if (!cData.empty())
            {
                for (const auto &row : cData)
                {
                    for (const auto &element : row)
                    {
                        REQUIRE(element >= 0);
                    }
                }
            }           
        }

        SECTION("100 ROWS DATASET CONVERSION TO DOUBLE")
        {
            auto cData = matrixConverter<double>(data1);

            if (!cData.empty())
            {
                for (const auto &row : cData)
                {
                    for (const auto &element : row)
                    {
                        REQUIRE(element >= 0);
                    }
                }
            }    
        }

        SECTION("10 000 ROWS DATASET CONVERSION TO FLOAT")
        {
            auto cData = matrixConverter<float>(data2);

            if (!cData.empty())
            {
                for (const auto &row : cData)
                {
                    for (const auto &element : row)
                    {
                        REQUIRE(element >= 0);
                    }
                }
            }    
        }

        SECTION("10 000  ROWS DATASET CONVERSION TO DOUBLE")
        {
            auto cData = matrixConverter<double>(data2);

           if (!cData.empty())
            {
                for (const auto &row : cData)
                {
                    for (const auto &element : row)
                    {
                        REQUIRE(element >= 0);
                    }
                }
            }    
        }

        SECTION("100 000 ROWS DATASET CONVERSION TO FLOAT")
        {
            auto cData = matrixConverter<float>(data3);

            if (!cData.empty())
            {
                for (const auto &row : cData)
                {
                    for (const auto &element : row)
                    {
                        REQUIRE(element >= 0);
                    }
                }
            }    
        }

        SECTION("100 000 ROWS DATASET CONVERSION TO DOUBLE")
        {
            auto cData = matrixConverter<double>(data3);

            if (!cData.empty())
            {
                for (const auto &row : cData)
                {
                    for (const auto &element : row)
                    {
                        REQUIRE(element >= 0);
                    }
                }
            }    
        }
    }

    SECTION("FIND METHODS")
    {
        auto data1 = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
        auto data2 = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
        auto data3 = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");

        auto cData2 = matrixConverter<float>(data2);
        auto cData3 = matrixConverter<double>(data3);

        REQUIRE(!data1.empty()); 
        REQUIRE(!cData2.empty()); 
        REQUIRE(!cData3.empty());

        SECTION("FIND METHOD")
        {
            // In find method, only returns 0 in both values if nothing was found 
            auto pos1 = find<std::string>(data1, "http://www.hatfield-saunders.net/");
            auto pos2 = find<float>(cData2, 2020.0F);
            auto pos3 = find<double>(cData3, 2021.0);

            REQUIRE(pos1[0] != 0);
            REQUIRE(pos1[1] != 0);
            REQUIRE(pos2[0] != 0);
            REQUIRE(pos2[1] != 0);
            REQUIRE(pos3[0] != 0);
            REQUIRE(pos3[1] != 0);
        }

        SECTION("FIND BY POSITION METHOD")
        {
            // Create vector to hold positions to search
            std::vector<int> pos1 {44, 4};
            std::vector<int> pos2 {5555, 5};
            std::vector<int> pos3 {66666, 6};

            // Return element from method
            auto el1 = findByPos<std::string>(data1, pos1);
            auto el2 = findByPos<float>(cData2, pos2);
            auto el3 = findByPos<double>(cData3, pos3);

            REQUIRE(!el1.empty());
            REQUIRE(el2 >= 0);
            REQUIRE(el3 >= 0);
        }

        SECTION("FIND ALL METHODS")
        {
            // In find method, matrix returns empty if nothing was found 
            auto pos1 = findAll<std::string>(data1, "Netherlands");
            auto pos2 = findAll<float>(cData2, 2020.0F);
            auto pos3 = findAll<double>(cData3, 2021.0);

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
        SECTION("100 ROWS READ CSV BENCHMARK")
        {
            auto start = startBenchmark();
            auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
            auto stop = stopBenchmark();

            REQUIRE(data.size() > 0);
            std::cout << "100 ROW DATABASE READ CSV - STRING" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("10 000 ROWS READ CSV BENCHMARK - STRING")
        {
            auto start = startBenchmark();
            auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
            auto stop = stopBenchmark();

            REQUIRE(data.size() > 0);
            std::cout << "10 000 ROW DATABASE READ CSV" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        SECTION("100 000 ROWS READ CSV BENCHMARK ")
        {
            auto start = startBenchmark();
            auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
            auto stop = stopBenchmark();

            REQUIRE(data.size() > 0);
            std::cout << "100 000 ROW DATABASE READ CSV" << std::endl;
            std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

    }

    SECTION("MATRIX CONVERSION METHODS BENCHMARKS")
    {
        auto data1 = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
        auto data2 = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
        auto data3 = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv"); 

        REQUIRE(data1.size() > 0); 
        REQUIRE(data2.size() > 0); 
        REQUIRE(data3.size() > 0); 

        SECTION("100 ROWS DATASET CONVERSION TO FLOAT")
        {
            auto start = startBenchmark();
            auto cData = matrixConverter<float>(data1);
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
            auto cData = matrixConverter<double>(data1);
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
            auto cData = matrixConverter<float>(data2);
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
            auto cData = matrixConverter<double>(data2);
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
            auto cData = matrixConverter<float>(data3);
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
            auto cData = matrixConverter<double>(data3);
            auto stop = stopBenchmark();

            if (!cData.empty())
            {
                std::cout << "100 000 ROW DATABASE CONVERTO TO DOUBLE BENCHMARK" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }
        }
    }

    SECTION("ASSORTED FIND METHODS BENCHMARKS")
    {
        SECTION("FIND METHOD BENCHMARKS")
        {
            SECTION("100 ROWS FIND METHOD BENCHMARK - STRING")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = find<std::string>(data, "http://www.hatfield-saunders.net/");
                auto stop = stopBenchmark();

                CHECK(pos[0] != 0);
                CHECK(pos[1] != 0);

                std::cout << "100 ROW DATABASE FIND" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 ROWS FIND METHOD BENCHMARK - FLOAT")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
                auto cData = matrixConverter<float>(data);

                REQUIRE(cData.size() > 0);

                auto start = startBenchmark();
                auto pos = find<float>(cData, 783.639F);
                auto stop = stopBenchmark();

                CHECK(pos[0] != 0);
                CHECK(pos[1] != 0);

                std::cout << "100 ROW DATABASE FIND - FLOAT" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 ROWS FIND METHOD BENCHMARK - DOUBLE")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
                auto cData = matrixConverter<double>(data);

                REQUIRE(cData.size() > 0);

                auto start = startBenchmark();
                auto pos = find<double>(cData, 783.639);
                auto stop = stopBenchmark();

                CHECK(pos[0] != 0);
                CHECK(pos[1] != 0);

                std::cout << "100 ROW DATABASE FIND - DOUBLE" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("10 000 ROWS FIND METHOD BENCHMARK - STRING")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = find<std::string>(data, "http://nixon.net/");
                auto stop = stopBenchmark();

                CHECK(pos[0] != 0);
                CHECK(pos[1] != 0);

                std::cout << "10 000 ROW DATABASE FIND - STRING" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("10 000 ROWS FIND METHOD BENCHMARK - FLOAT")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
                auto cData = matrixConverter<float>(data);

                REQUIRE(cData.size() > 0);

                auto start = startBenchmark();
                auto pos = find<float>(cData, 423.632F);
                auto stop = stopBenchmark();

                CHECK(pos[0] != 0);
                CHECK(pos[1] != 0);

                std::cout << "10 000 ROW DATABASE FIND - FLOAT" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("10 000 ROWS FIND METHOD BENCHMARK - DOUBLE")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
                auto cData = matrixConverter<double>(data);

                REQUIRE(cData.size() > 0);

                auto start = startBenchmark();
                auto pos = find<double>(cData, 423.632);
                auto stop = stopBenchmark();

                CHECK(pos[0] != 0);
                CHECK(pos[1] != 0);

                std::cout << "10 000 ROW DATABASE FIND - DOUBLE" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 000 ROWS FIND METHOD BENCHMARK - STRING")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = find<std::string>(data, "https://www.walter.com/");
                auto stop = stopBenchmark();

                CHECK(pos[0] != 0);
                CHECK(pos[1] != 0);

                std::cout << "100 000 ROW DATABASE FIND - STRING" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 000 ROWS FIND METHOD BENCHMARK - FLOAT")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
                auto cData = matrixConverter<float>(data);

                REQUIRE(cData.size() > 0);

                auto start = startBenchmark();
                auto pos = find<float>(cData, 157.189F);
                auto stop = stopBenchmark();

                CHECK(pos[0] != 0);
                CHECK(pos[1] != 0);

                std::cout << "100 000 ROW DATABASE FIND - FLOAT" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 000 ROWS FIND METHOD BENCHMARK - DOUBLE")
            {
                // For testing purposes, the desired element will always be the last possible
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
                auto cData = matrixConverter<double>(data);

                REQUIRE(cData.size() > 0);

                auto start = startBenchmark();
                auto pos = find<double>(cData, 157.189);
                auto stop = stopBenchmark();

                CHECK(pos[0] != 0);
                CHECK(pos[1] != 0);

                std::cout << "100 000 ROW DATABASE FIND - DOUBLE" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }
           
        }

        SECTION("FIND BY POSITION METHOD BENCHMARKS")
        {
            SECTION("100 ROWS FIND BY POSITION METHOD - STRING")
            {
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
                std::vector<int> pos {44, 4};

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto el = findByPos<std::string>(data, pos);
                auto stop = stopBenchmark();

                REQUIRE(!el.empty());

                std::cout << "100 ROW DATABASE FIND BY POSITION - STRING" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 ROWS FIND BY POSITION METHOD - FLOAT")
            {
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
                auto cData = matrixConverter<float>(data);
                std::vector<int> pos {55, 5};

                REQUIRE(cData.size() > 0);

                auto start = startBenchmark();
                auto el = findByPos<float>(cData, pos);
                auto stop = stopBenchmark();

                REQUIRE(el >= 0);

                std::cout << "100 ROW DATABASE FIND BY POSITION - FLOAT" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 ROWS FIND BY POSITION METHOD - DOUBLE")
            {
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
                auto cData = matrixConverter<double>(data);
                std::vector<int> pos {66, 6};

                REQUIRE(cData.size() > 0);

                auto start = startBenchmark();
                auto el = findByPos<double>(cData, pos);
                auto stop = stopBenchmark();

                REQUIRE(el >= 0);

                std::cout << "100 ROW DATABASE FIND BY POSITION - DOUBLE" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("10 000 ROWS FIND BY POSITION METHOD - STRING")
            {
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
                std::vector<int> pos {4444, 4};

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto el = findByPos<std::string>(data, pos);
                auto stop = stopBenchmark();

                REQUIRE(!el.empty());

                std::cout << "10 000 ROW DATABASE FIND BY POSITION - STRING" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("10 000 ROWS FIND BY POSITION METHOD - FLOAT")
            {
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
                auto cData = matrixConverter<float>(data);
                std::vector<int> pos {5555, 5};

                REQUIRE(cData.size() > 0);

                auto start = startBenchmark();
                auto el = findByPos<float>(cData, pos);
                auto stop = stopBenchmark();

                REQUIRE(el >= 0);

                std::cout << "10 000 ROW DATABASE FIND BY POSITION - FLOAT" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("10 000 ROWS FIND BY POSITION METHOD - DOUBLE")
            {
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
                auto cData = matrixConverter<double>(data);
                std::vector<int> pos {6666, 6};

                REQUIRE(cData.size() > 0);

                auto start = startBenchmark();
                auto el = findByPos<double>(cData, pos);
                auto stop = stopBenchmark();

                REQUIRE(el >= 0);

                std::cout << "10 000 ROW DATABASE FIND BY POSITION - DOUBLE" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 000 ROWS FIND BY POSITION METHOD - STRING")
            {
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
                std::vector<int> pos {44444, 4};

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto el = findByPos<std::string>(data, pos);
                auto stop = stopBenchmark();

                REQUIRE(!el.empty());

                std::cout << "100 000 ROW DATABASE FIND BY POSITION - STRING" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 000 ROWS FIND BY POSITION METHOD - FLOAT")
            {
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
                auto cData = matrixConverter<float>(data);
                std::vector<int> pos {55555, 5};

                REQUIRE(cData.size() > 0);

                auto start = startBenchmark();
                auto el = findByPos<float>(cData, pos);
                auto stop = stopBenchmark();

                REQUIRE(el >= 0);

                std::cout << "100 000 ROW DATABASE FIND BY POSITION - FLOAT" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 000 ROWS FIND BY POSITION METHOD - DOUBLE")
            {
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
                auto cData = matrixConverter<double>(data);
                std::vector<int> pos {66666, 6};

                REQUIRE(cData.size() > 0);

                auto start = startBenchmark();
                auto el = findByPos<double>(cData, pos);
                auto stop = stopBenchmark();

                REQUIRE(el >= 0);

                std::cout << "100 000 ROW DATABASE FIND BY POSITION - DOUBLE" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }
        }

        SECTION("FIND ALL METHOD BENCHMARKS")
        {
            SECTION("100 ROWS FIND ALL METHOD BENCHMARK - STRING")
            {
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = findAll<std::string>(data, "Netherlands");
                auto stop = stopBenchmark();

                CHECK(!pos.empty());

                std::cout << "100 ROW DATABASE FIND ALL - STRING" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 ROWS FIND ALL METHOD BENCHMARK - FLOAT")
            {
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
                auto cData = matrixConverter<float>(data);

                REQUIRE(cData.size() > 0);

                auto start = startBenchmark();
                auto pos = findAll<float>(cData, 2020.0F);
                auto stop = stopBenchmark();

                CHECK(!pos.empty());

                std::cout << "100 ROW DATABASE FIND ALL - FLOAT" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 ROWS FIND ALL METHOD BENCHMARK - DOUBLE")
            {
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100.csv");
                auto cData = matrixConverter<double>(data);

                REQUIRE(cData.size() > 0);

                auto start = startBenchmark();
                auto pos = findAll<double>(cData, 2021.0);
                auto stop = stopBenchmark();

                CHECK(!pos.empty());

                std::cout << "100 ROW DATABASE FIND ALL - DOUBLE" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("10 000 ROWS FIND ALL METHOD BENCHMARK - STRING")
            {
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = findAll<std::string>(data, "Netherlands");
                auto stop = stopBenchmark();

                CHECK(!pos.empty());

                std::cout << "10 000 ROW DATABASE FIND ALL - STRING" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("10 000 ROWS FIND ALL METHOD BENCHMARK - FLOAT")
            {
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
                auto cData = matrixConverter<float>(data);

                REQUIRE(cData.size() > 0);

                auto start = startBenchmark();
                auto pos = findAll<float>(cData, 2020.0F);
                auto stop = stopBenchmark();

                CHECK(!pos.empty());

                std::cout << "10 000 ROW DATABASE FIND ALL - FLOAT" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("10 000 ROWS FIND ALL METHOD BENCHMARK - DOUBLE")
            {
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-10000.csv");
                auto cData = matrixConverter<double>(data);

                REQUIRE(cData.size() > 0);

                auto start = startBenchmark();
                auto pos = findAll<double>(cData, 2021.0);
                auto stop = stopBenchmark();

                CHECK(!pos.empty());

                std::cout << "10 000 ROW DATABASE FIND ALL - DOUBLE" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 000 ROWS FIND ALL METHOD BENCHMARK - STRING")
            {
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");

                REQUIRE(data.size() > 0);

                auto start = startBenchmark();
                auto pos = findAll<std::string>(data, "Netherlands");
                auto stop = stopBenchmark();

                CHECK(!pos.empty());

                std::cout << "100 000 ROW DATABASE FIND ALL - STRING" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 000 ROWS FIND ALL METHOD BENCHMARK - FLOAT")
            {
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
                auto cData = matrixConverter<float>(data);

                REQUIRE(cData.size() > 0);

                auto start = startBenchmark();
                auto pos = findAll<float>(cData, 2020.0F);
                auto stop = stopBenchmark();

                CHECK(!pos.empty());

                std::cout << "100 000 ROW DATABASE FIND ALL - FLOAT" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }

            SECTION("100 000 ROWS FIND ALL METHOD BENCHMARK - DOUBLE")
            {
                auto data = readCSV("/home/muzaodamassa/MLPP/tests/Datasets/customers-100000.csv");
                auto cData = matrixConverter<double>(data);

                REQUIRE(cData.size() > 0);

                auto start = startBenchmark();
                auto pos = findAll<double>(cData, 2021.0);
                auto stop = stopBenchmark();

                CHECK(!pos.empty());

                std::cout << "100 000 ROW DATABASE FIND ALL - DOUBLE" << std::endl;
                std::cout << getDuration(start, stop, Nanoseconds) << std::endl;
                std::cout << "NUMBER OF TIMES ELEMENT WAS FOUND: " << pos.size() << std::endl;
                std::cout << "-----------------------------------------------" << std::endl;
            }
        }
     
    }
}
        