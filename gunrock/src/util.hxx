#pragma once
#include "types.hxx"
#include <cstdarg>
#include <string>
#include <sstream>
#include <map>

BEGIN_GUNROCK_NAMESPACE

namespace detail {

inline std::string stringprintf(const char* format, ...) {
  va_list args;
  va_start(args, format);
  int len = vsnprintf(0, 0, format, args);
  va_end(args);

  // allocate space.
  std::string text;
  text.resize(len);

  va_start(args, format);
  vsnprintf(&text[0], len + 1, format, args);
  va_end(args);

  return text;
}

} // namespace detail

/******************************************************************************
 * Command-line parsing functionality
 ******************************************************************************/

/**
 * CommandLineArgs interface
 */
class CommandLineArgs
{
private:
    int argc;
    char ** argv;
protected:

    std::map<std::string, std::string> pairs;

public:

    // Constructor
    CommandLineArgs(int _argc, char **_argv) : argc(_argc), argv(_argv)
    {
        for (int i = 1; i < argc; i++)
        {
            std::string arg = argv[i];

            if ((arg[0] != '-') || (arg[1] != '-'))
            {
                continue;
            }

            std::string::size_type pos;
            std::string key, val;
            if ((pos = arg.find('=')) == std::string::npos)
            {
                key = std::string(arg, 2, arg.length() - 2);
                val = "";
            }
            else
            {
                key = std::string(arg, 2, pos - 2);
                val = std::string(arg, pos + 1, arg.length() - 1);
            }
            pairs[key] = val;
        }
    }

    // Checks whether a flag "--<flag>" is present in the commandline
    bool CheckCmdLineFlag(const char* arg_name)
    {
        std::map<std::string, std::string>::iterator itr;
        if ((itr = pairs.find(arg_name)) != pairs.end())
        {
            return true;
        }
        return false;
    }

    // Returns the value specified for a given commandline
    // parameter --<flag>=<value>
    template <typename T>
    void GetCmdLineArgument(const char *arg_name, T &val);

    // Returns the values specified for a given commandline
    // parameter --<flag>=<value>,<value>*
    template <typename T>
    void GetCmdLineArguments(const char *arg_name, std::vector<T> &vals);

    // The number of pairs parsed
    int ParsedArgc()
    {
        return pairs.size();
    }

    std::string GetEntireCommandLine() const
    {
        std::string commandLineStr = "";
        for (int i = 0; i < argc; i++)
        {
            commandLineStr.append(std::string(argv[i]).append((i < argc - 1) ? " " : ""));
        }
        return commandLineStr;
    }

    template <typename T>
    void ParseArgument(const char *name, T &val)
    {
        if (CheckCmdLineFlag(name))
        {
            GetCmdLineArgument(name, val);
        }
    }
};

template <typename T>
void CommandLineArgs::GetCmdLineArgument(
    const char *arg_name,
    T &val)
{
    std::map<std::string, std::string>::iterator itr;
    if ((itr = pairs.find(arg_name)) != pairs.end())
    {
        std::istringstream str_stream(itr->second);
        str_stream >> val;
    }
}

template <typename T>
void CommandLineArgs::GetCmdLineArguments(
    const char *arg_name,
    std::vector<T> &vals)
{
    // Recover multi-value string
    std::map<std::string, std::string>::iterator itr;
    if ((itr = pairs.find(arg_name)) != pairs.end())
    {

        // Clear any default values
        vals.clear();

        std::string val_string = itr->second;
        std::istringstream str_stream(val_string);
        std::string::size_type old_pos = 0;
        std::string::size_type new_pos = 0;

        // Iterate comma-separated values
        T val;
        while ((new_pos = val_string.find(',', old_pos)) != std::string::npos)
        {

            if (new_pos != old_pos)
            {
                str_stream.width(new_pos - old_pos);
                str_stream >> val;
                vals.push_back(val);
            }

            // skip over comma
            str_stream.ignore(1);
            old_pos = new_pos + 1;
        }

        // Read last value
        str_stream >> val;
        vals.push_back(val);
    }
}

END_GUNROCK_NAMESPACE
