#include <gtest/gtest.h>

#include "robustlearn_control/system_check_validation.hpp"

TEST(SystemCheckValidation, AcceptsPositiveSteps)
{
  EXPECT_TRUE(
    robustlearn_control::is_valid_total_steps(1));

  EXPECT_TRUE(
    robustlearn_control::is_valid_total_steps(5));

  EXPECT_TRUE(
    robustlearn_control::is_valid_total_steps(100));
}

TEST(SystemCheckValidation, RejectsZeroSteps)
{
  EXPECT_FALSE(
    robustlearn_control::is_valid_total_steps(0));
}

TEST(SystemCheckValidation, RejectsNegativeSteps)
{
  EXPECT_FALSE(
    robustlearn_control::is_valid_total_steps(-1));

  EXPECT_FALSE(
    robustlearn_control::is_valid_total_steps(-100));
}
