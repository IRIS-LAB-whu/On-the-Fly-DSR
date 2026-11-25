#include <boost/config.hpp>

#if defined(BOOST_NO_EXCEPTIONS)

#include <boost/assert/source_location.hpp>

#include <exception>

namespace boost {

void throw_exception(const std::exception&)
{
    std::terminate();
}

void throw_exception(const std::exception&, const boost::source_location&)
{
    std::terminate();
}

}  // namespace boost

#endif
