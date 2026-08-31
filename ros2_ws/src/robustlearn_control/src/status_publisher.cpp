#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class StatusPublisher : public rclcpp::Node
{
public:
  StatusPublisher()
  : Node("status_publisher"), count_(0)
  {
    message_prefix_ = this->declare_parameter<std::string>(
      "message_prefix", "RobustLearn status message ");

    const auto publish_period_ms = this->declare_parameter<int64_t>(
      "publish_period_ms", 1000);

    if (publish_period_ms <= 0) {
      throw std::invalid_argument("publish_period_ms must be greater than zero");
    }

    publisher_ = this->create_publisher<std_msgs::msg::String>(
      "system_status_broken", 10);

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(publish_period_ms),
      std::bind(&StatusPublisher::publish_status, this));
  }

private:
  void publish_status()
  {
    std_msgs::msg::String message;
    message.data = message_prefix_ + std::to_string(count_++);

    RCLCPP_INFO(
      this->get_logger(),
      "Publishing: '%s'",
      message.data.c_str());

    publisher_->publish(message);
  }

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::string message_prefix_;
  std::size_t count_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::spin(std::make_shared<StatusPublisher>());

  rclcpp::shutdown();
  return 0;
}
