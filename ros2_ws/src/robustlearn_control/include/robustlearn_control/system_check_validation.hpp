#pragma once

#include <cstdint>

namespace robustlearn_control
{

inline bool is_valid_total_steps(std::int32_t total_steps)
{
  return total_steps > 0;
}

}  // namespace robustlearn_control
