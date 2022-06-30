#pragma once
#include <fmt/format.h>
#include <getopt.h>
#include <sstream>
#include <sys/time.h>

namespace DArray {

    void test_cuda();

    inline long long time_since_epoch_in_ms() {
        using namespace std::chrono;
        return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    }


    class ArgsParser {
    public:
        ArgsParser()
        {
            add_option(m_show_help, "Display this message", "help", 0);
        }

        enum class Required {
            Yes,
            No
        };

        struct Option {
            bool requires_argument { true };
            const char* help_string { nullptr };
            const char* long_name { nullptr };
            char short_name { 0 };
            const char* value_name { nullptr };
            std::function<bool(const char*)> accept_value;

            std::string name_for_display() const
            {
                if (long_name)
                    return fmt::format("--{}", long_name);
                return fmt::format("-{}", short_name);
            }
        };

        struct Arg {
            const char* help_string { nullptr };
            const char* name { nullptr };
            int min_values { 0 };
            int max_values { 1 };
            std::function<bool(const char*)> accept_value;
        };

//        bool parse(int argc, char** argv, bool exit_on_failure = true);
        bool parse(int argc, char** argv, bool exit_on_failure = true)
        {
            auto print_usage_and_exit = [this, argv, exit_on_failure] {
                print_usage(stderr, argv[0]);
                if (exit_on_failure)
                    exit(1);
            };

            std::vector<option> long_options;
            std::stringstream short_options_builder;

            int index_of_found_long_option = -1;

            // Tell getopt() to reset its internal state, and start scanning from optind = 1.
            // We could also set optreset = 1, but the host platform may not support that.
            optind = 0;

            for (size_t i = 0; i < m_options.size(); i++) {
                auto& opt = m_options[i];
                if (opt.long_name) {
                    option long_opt {
                            opt.long_name,
                            opt.requires_argument ? required_argument : no_argument,
                            &index_of_found_long_option,
                            static_cast<int>(i)
                    };
                    long_options.push_back(long_opt);
                }
                if (opt.short_name) {
//                    short_options_builder.append(opt.short_name);
                    short_options_builder << opt.short_name;
                    if (opt.requires_argument)
//                        short_options_builder.append(':');
                        short_options_builder << ':';
                }
            }
            long_options.push_back({ 0, 0, 0, 0 });

            std::string short_options = short_options_builder.str();

            while (true) {
                int c = getopt_long(argc, argv, short_options.c_str(), long_options.data(), nullptr);
                if (c == -1) {
                    // We have reached the end.
                    break;
                } else if (c == '?') {
                    // There was an error, and getopt() has already
                    // printed its error message.
                    print_usage_and_exit();
                    return false;
                }

                // Let's see what option we just found.
                Option* found_option = nullptr;
                if (c == 0) {
                    // It was a long option.
                    assert(index_of_found_long_option >= 0);
                    found_option = &m_options[index_of_found_long_option];
                    index_of_found_long_option = -1;
                } else {
                    // It was a short option, look it up.
//                    auto it = m_options.find_if([c](auto& opt) { return c == opt.short_name; });
                    auto it = find_if(m_options.begin(), m_options.end(), [c](auto& opt) { return c == opt.short_name; });
                    assert(it != m_options.end());

                    found_option = &*it;
                }
                assert(found_option);

                const char* arg = found_option->requires_argument ? optarg : nullptr;
                if (!found_option->accept_value(arg)) {
                    fmt::print("\033[31mInvalid value for option \033[1m{}\033[22m, dude\033[0m", found_option->name_for_display());
                    print_usage_and_exit();
                    return false;
                }
            }

            // We're done processing options, now let's parse positional arguments.

            int values_left = argc - optind;
            int num_values_for_arg[m_positional_args.size()];
            int total_values_required = 0;
            for (size_t i = 0; i < m_positional_args.size(); i++) {
                auto& arg = m_positional_args[i];
                num_values_for_arg[i] = arg.min_values;
                total_values_required += arg.min_values;
            }

            if (total_values_required > values_left) {
                print_usage_and_exit();
                return false;
            }
            int extra_values_to_distribute = values_left - total_values_required;

            for (size_t i = 0; i < m_positional_args.size(); i++) {
                auto& arg = m_positional_args[i];
                int extra_values_to_this_arg = std::min(arg.max_values - arg.min_values, extra_values_to_distribute);
                num_values_for_arg[i] += extra_values_to_this_arg;
                extra_values_to_distribute -= extra_values_to_this_arg;
                if (extra_values_to_distribute == 0)
                    break;
            }

            if (extra_values_to_distribute > 0) {
                // We still have too many values :(
                print_usage_and_exit();
                return false;
            }

            for (size_t i = 0; i < m_positional_args.size(); i++) {
                auto& arg = m_positional_args[i];
                for (int j = 0; j < num_values_for_arg[i]; j++) {
                    const char* value = argv[optind++];
                    if (!arg.accept_value(value)) {
                        fmt::print("Invalid value for argument {}", arg.name);
                        print_usage_and_exit();
                        return false;
                    }
                }
            }

            // We're done parsing! :)
            // Now let's show help if requested.
            if (m_show_help) {
                print_usage(stdout, argv[0]);
                if (exit_on_failure)
                    exit(0);
                return false;
            }

            return true;
        }
        // *Without* trailing newline!
        void set_general_help(const char* help_string) { m_general_help = help_string; };
//        void print_usage(FILE*, const char* argv0);

//        void add_option(Option&&);
//        void add_option(bool& value, const char* help_string, const char* long_name, char short_name);
        void add_option(Option&& option)
        {
            m_options.push_back(std::move(option));
        }
        void add_option(bool& value, const char* help_string, const char* long_name, char short_name)
        {
            Option option {
                    false,
                    help_string,
                    long_name,
                    short_name,
                    nullptr,
                    [&value](const char* s) {
//                        assert(s != nullptr);
                        value = true;
                        return true;
                    }
            };
            add_option(std::move(option));
        }
//        void add_option(const char*& value, const char* help_string, const char* long_name, char short_name, const char* value_name);
        void add_option(const char*& value, const char* help_string, const char* long_name, char short_name, const char* value_name)
        {
            Option option {
                    true,
                    help_string,
                    long_name,
                    short_name,
                    value_name,
                    [&value](const char* s) {
                        value = s;
                        return true;
                    }
            };
            add_option(std::move(option));
        }
//        void add_option(int& value, const char* help_string, const char* long_name, char short_name, const char* value_name);
        void add_option(int& value, const char* help_string, const char* long_name, char short_name, const char* value_name)
        {
            Option option {
                    true,
                    help_string,
                    long_name,
                    short_name,
                    value_name,
                    [&value](const char* s) {
                        value = atoi(s);
                        return true;
                    }
            };
            add_option(std::move(option));
        }
//        void add_option(double& value, const char* help_string, const char* long_name, char short_name, const char* value_name);
        static constexpr bool isnan(double __x) { return __builtin_isnan(__x); }
        static double convert_to_double(const char* s) {
            char* p;
            double v = strtod(s, &p);
            if (isnan(v) || p == s)
                return 0;
            return v;
        }
        void add_option(double& value, const char* help_string, const char* long_name, char short_name, const char* value_name)
        {
            Option option {
                    true,
                    help_string,
                    long_name,
                    short_name,
                    value_name,
                    [&value](const char* s) {
                        value = convert_to_double(s);
                        return true;
                    }
            };
            add_option(std::move(option));
        }

        void add_positional_argument(Arg&& arg)
        {
            m_positional_args.push_back(std::move(arg));
        }
        void add_positional_argument(const char*& value, const char* help_string, const char* name, Required required)
        {
            Arg arg {
                    help_string,
                    name,
                    required == Required::Yes ? 1 : 0,
                    1,
                    [&value](const char* s) {
                        value = s;
                        return true;
                    }
            };
            add_positional_argument(std::move(arg));
        }

        void add_positional_argument(int& value, const char* help_string, const char* name, Required required)
        {
            Arg arg {
                    help_string,
                    name,
                    required == Required::Yes ? 1 : 0,
                    1,
                    [&value](const char* s) {
                        value = atoi(s);
                        return true;
                    }
            };
            add_positional_argument(std::move(arg));
        }

        void add_positional_argument(double& value, const char* help_string, const char* name, Required required)
        {
            Arg arg {
                    help_string,
                    name,
                    required == Required::Yes ? 1 : 0,
                    1,
                    [&value](const char* s) {
                        value = convert_to_double(s);
                        return true;
                    }
            };
            add_positional_argument(std::move(arg));
        }

        void add_positional_argument(std::vector<const char*>& values, const char* help_string, const char* name, Required required)
        {
            Arg arg {
                    help_string,
                    name,
                    required == Required::Yes ? 1 : 0,
                    INT_MAX,
                    [&values](const char* s) {
                        values.push_back(s);
                        return true;
                    }
            };
            add_positional_argument(std::move(arg));
        }

        void print_usage(FILE* file, const char* argv0)
        {
            fmt::print(file, "Usage:\n\t\033[1m{}\033[0m", argv0);

            for (auto& opt : m_options) {
                if (opt.long_name && !strcmp(opt.long_name, "help"))
                    continue;
                if (opt.requires_argument)
                    fmt::print(file, " [{} {}]", opt.name_for_display(), opt.value_name);
                else
                    fmt::print(file, " [{}]", opt.name_for_display());
            }
            for (auto& arg : m_positional_args) {
                bool required = arg.min_values > 0;
                bool repeated = arg.max_values > 1;

                if (required && repeated)
                    fmt::print(file, " <{}...>", arg.name);
                else if (required && !repeated)
                    fmt::print(file, " <{}>", arg.name);
                else if (!required && repeated)
                    fmt::print(file, " [{}...]", arg.name);
                else if (!required && !repeated)
                    fmt::print(file, " [{}]", arg.name);
            }
            fmt::print(file,"\n");

            if (m_general_help != nullptr && m_general_help[0] != '\0') {
                fmt::print(file, "\nDescription:\n");
                fmt::print(file, "{}\n", m_general_help);
            }

            if (!m_options.empty())
                fmt::print(file, "\nOptions:\n");
            for (auto& opt : m_options) {
                auto print_argument = [&]() {
                    if (opt.value_name) {
                        if (opt.requires_argument)
                            fmt::print(file, " {}", opt.value_name);
                        else
                            fmt::print(file, " [{}]", opt.value_name);
                    }
                };
                fmt::print(file, "\t");
                if (opt.short_name) {
                    fmt::print(file, "\033[1m-{}\033[0m", opt.short_name);
                    print_argument();
                }
                if (opt.short_name && opt.long_name)
                    fmt::print(file, ", ");
                if (opt.long_name) {
                    fmt::print(file, "\033[1m--{}\033[0m", opt.long_name);
                    print_argument();
                }

                if (opt.help_string)
                    fmt::print(file, "\t{}", opt.help_string);
                fmt::print(file, "\n");
            }

            if (!m_positional_args.empty())
                fmt::print(file, "\nArguments:\n");

            for (auto& arg : m_positional_args) {
                fmt::print(file, "\t\033[1m{}\033[0m", arg.name);
                if (arg.help_string)
                    fmt::print(file, "\t{}", arg.help_string);
                fmt::print(file, "\n");
            }
        }

    private:
        std::vector<Option> m_options;
        std::vector<Arg> m_positional_args;

        bool m_show_help { false };
        const char* m_general_help { nullptr };
    };

    class ElapsedTimer {
    public:
        ElapsedTimer(bool precise = false)
                : m_precise(precise)
        {
        }

        bool is_valid() const { return m_valid; }
//        void start();
//        int elapsed() const;

        const struct timeval& origin_time() const { return m_origin_time; }

        template<typename TimevalType>
        inline static void timeval_sub(const TimevalType& a, const TimevalType& b, TimevalType& result)
        {
            result.tv_sec = a.tv_sec - b.tv_sec;
            result.tv_usec = a.tv_usec - b.tv_usec;
            if (result.tv_usec < 0) {
                --result.tv_sec;
                result.tv_usec += 1'000'000;
            }
        }

        void start()
        {
            m_valid = true;
            timespec now_spec;
            clock_gettime(m_precise ? CLOCK_MONOTONIC : CLOCK_MONOTONIC_COARSE, &now_spec);
            m_origin_time.tv_sec = now_spec.tv_sec;
            m_origin_time.tv_usec = now_spec.tv_nsec / 1000;
        }

        int elapsed() const
        {
            assert(is_valid());
            struct timeval now;
            timespec now_spec;
            clock_gettime(m_precise ? CLOCK_MONOTONIC : CLOCK_MONOTONIC_COARSE, &now_spec);
            now.tv_sec = now_spec.tv_sec;
            now.tv_usec = now_spec.tv_nsec / 1000;
            struct timeval diff;
            timeval_sub(now, m_origin_time, diff);
            return diff.tv_sec * 1000 + diff.tv_usec / 1000;
        }



    private:
        bool m_precise { false };
        bool m_valid { false };
        struct timeval m_origin_time {
                0, 0
        };
    };


    struct Tracer {
        enum class EventType {StartEvent, EndEvent};


        long long start;
        long long  end;
        const char *op;
        bool standalone {false};

        static bool is_trace_on;
        static long long time_origin;

        struct Event {
            EventType etype;
            const char* op ;
            long start;
            long end;
        };

        static std::vector<Event> events;

        void clear_events() {
            events.clear();
        }

        static void trace_on() {
            time_origin = time_since_epoch_in_ms();
            is_trace_on = true;
        }

        static void trace_off() {
            is_trace_on = false;
        }


        Tracer(const char* op_) : end{-1}, op{op_} {
            if (is_trace_on) {
                start = time_since_epoch_in_ms() - time_origin;
                events.push_back({EventType::StartEvent, op, start, end});
//                fmt::print("{} tracer start at {}\n", op, start);
            }
        }
        Tracer(const char* op_, bool standalone_) : end{-1}, op{op_}, standalone{standalone_} {
                start = time_since_epoch_in_ms();
//                fmt::print("{} tracer start at {}\n", op, start);
        }
        ~Tracer() {
            if(is_trace_on) {
                end = time_since_epoch_in_ms() - time_origin;
                events.push_back({EventType::EndEvent, op, start, end});
            }
            if(standalone) {
                end = time_since_epoch_in_ms();
                fmt::print("{} tracer end at {}; started {}, takes {} (ms)\n", op, end, start, end-start);
            }
        }

        static void save_events_to_file(std::string filename) {
            FILE *f = fopen(filename.c_str(), "w");
            for(auto &event : events) {
                if(event.etype==Tracer::EventType::StartEvent)
                    fmt::print(f, "{} START \"{}\"\n", event.start, event.op);
                else
                    fmt::print(f, "{} END \"{}\"\n", event.end, event.op);
            }
            fclose(f);
        }


    };
    std::vector<DArray::Tracer::Event> DArray::Tracer::events {};
    bool DArray::Tracer::is_trace_on {false};
    long long DArray::Tracer::time_origin {0};

    double relative_difference(double a, double b) {
        return abs(a-b)/b;
    }

    /*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 */

#if defined(_WIN32)
    #include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#include <stdio.h>

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif





/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
    size_t getPeakRSS( )
    {
#if defined(_WIN32)
        /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo( GetCurrentProcess( ), &info, sizeof(info) );
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
        /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ( (fd = open( "/proc/self/psinfo", O_RDONLY )) == -1 )
        return (size_t)0L;      /* Can't open? */
    if ( read( fd, &psinfo, sizeof(psinfo) ) != sizeof(psinfo) )
    {
        close( fd );
        return (size_t)0L;      /* Can't read? */
    }
    close( fd );
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
        /* BSD, Linux, and OSX -------------------------------------- */
        struct rusage rusage;
        getrusage( RUSAGE_SELF, &rusage );
#if defined(__APPLE__) && defined(__MACH__)
        return (size_t)rusage.ru_maxrss;
#else
        return (size_t)(rusage.ru_maxrss * 1024L);
#endif

#else
        /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
    }





/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
    size_t getCurrentRSS( )
    {
#if defined(_WIN32)
        /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo( GetCurrentProcess( ), &info, sizeof(info) );
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
        /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if ( task_info( mach_task_self( ), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount ) != KERN_SUCCESS )
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
        /* Linux ---------------------------------------------------- */
        long rss = 0L;
        FILE* fp = NULL;
        if ( (fp = fopen( "/proc/self/statm", "r" )) == NULL )
            return (size_t)0L;      /* Can't open? */
        if ( fscanf( fp, "%*s%ld", &rss ) != 1 )
        {
            fclose( fp );
            return (size_t)0L;      /* Can't read? */
        }
        fclose( fp );
        return (size_t)rss * (size_t)sysconf( _SC_PAGESIZE);

#else
        /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
    }

#include <execinfo.h> // for backtrace
#include <dlfcn.h>    // for dladdr
#include <cxxabi.h>   // for __cxa_demangle

#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>

// This function produces a stack backtrace with demangled function & method names.
    std::string Backtrace(int skip = 1)
    {
        void *callstack[128];
        const int nMaxFrames = sizeof(callstack) / sizeof(callstack[0]);
        char buf[1024];
        int nFrames = backtrace(callstack, nMaxFrames);
        char **symbols = backtrace_symbols(callstack, nFrames);

        std::ostringstream trace_buf;
        for (int i = skip; i < nFrames; i++) {
            printf("%s\n", symbols[i]);

            Dl_info info;
            if (dladdr(callstack[i], &info) && info.dli_sname) {
                char *demangled = NULL;
                int status = -1;
                if (info.dli_sname[0] == '_')
                    demangled = abi::__cxa_demangle(info.dli_sname, NULL, 0, &status);
                snprintf(buf, sizeof(buf), "%-3d %*p %s + %zd\n",
                         i, int(2 + sizeof(void*) * 2), callstack[i],
                         status == 0 ? demangled :
                         info.dli_sname == 0 ? symbols[i] : info.dli_sname,
                         (char *)callstack[i] - (char *)info.dli_saddr);
                free(demangled);
            } else {
                snprintf(buf, sizeof(buf), "%-3d %*p %s\n",
                         i, int(2 + sizeof(void*) * 2), callstack[i], symbols[i]);
            }
            trace_buf << buf;
        }
        free(symbols);
        if (nFrames == nMaxFrames)
            trace_buf << "[truncated]\n";
        return trace_buf.str();
    }
}