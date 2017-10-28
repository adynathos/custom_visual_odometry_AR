
#pragma once
#include <iostream>
#include <chrono>

class RunTime
{
public:
	RunTime(std::string const& msg = "Time: ")
	: start_timestamp(std::chrono::high_resolution_clock::now())
		, message(msg)
	{
	}

	void end()
	{
		std::chrono::duration<double, std::milli> duration_ms = std::chrono::high_resolution_clock::now() - start_timestamp;
		std::cout << message << duration_ms.count() << " ms" << std::endl;
	}

private:
	using dsec = std::chrono::duration<double, std::ratio<1> >;

	std::chrono::time_point<std::chrono::high_resolution_clock> start_timestamp;
	std::string message;
};
