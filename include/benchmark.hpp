#ifndef BENCHMARK_HPP_
#define BENCHMARK_HPP_

#include <stdexcept>
#include <sys/time.h>

class Timer
{

    private:
    double start_time;

    public:
    Timer() : start_time(0)
    {}

    void start()
    {
        struct timeval tval;
        gettimeofday(&tval, NULL);
        start_time = tval.tv_sec * 1000000 + tval.tv_usec;
    }

    double end() const
    {
        struct timeval tval;
        gettimeofday(&tval, NULL);
        double end_time = tval.tv_sec * 1000000 + tval.tv_usec;
        return static_cast<double>(end_time - start_time) / 1000000.0;
    }

};

#endif
