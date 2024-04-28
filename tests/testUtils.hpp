#pragma once

#include <chrono>
#include <iostream>

// Methods to help with testing 

namespace Utils
{
    // Subtract all elements of given vector 
    template <typename T> void subtractAllElements(std::vector<T>& vector, const int& toSubtract)
    {
        for (auto &element : vector)
        {
            element -= toSubtract;
        }
    }
}


namespace Benchmark
{
    enum TimeUnit 
    {
        Seconds,
        Microseconds, 
        Milliseconds,
        Nanoseconds
    };

    // Benchmarking
    std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::_V2::system_clock::duration> startBenchmark()
    {
        return std::chrono::high_resolution_clock::now();
    }

    std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::_V2::system_clock::duration> stopBenchmark()
    {
        return std::chrono::high_resolution_clock::now();
    }

    std::string getDuration(std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::_V2::system_clock::duration>& start,
        std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::_V2::system_clock::duration>& stop, 
            TimeUnit timeUnit = Microseconds)
    {
        if (timeUnit == Nanoseconds) {
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            double time = duration.count();
            return "Process duration in Nanoseconds: " + std::to_string(time);
        } 
        else if (timeUnit == Milliseconds) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            double time = duration.count();
            return "Process duration in Milliseconds: " + std::to_string(time);
        } 
        else if (timeUnit == Microseconds) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            double time = duration.count();
            return "Process duration in Microseconds: " + std::to_string(time);
        } 
        else {
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            double time = duration.count() / 1'000'000'000.0; // Convert nanoseconds to seconds
            return "Process duration in Seconds: " + std::to_string(time);
        }
    }

    
}