#include <memory>
#include <functional>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class StatusSubscriber : public rclcpp::Node
{
public:
  StatusSubscriber()
  : Node("status_subscriber")
  {
    subscription_ = this->create_subscription<std_msgs::msg::String>(
      "system_status",
      10,
      std::bind(
        &StatusSubscriber::status_callback,
        this,
        std::placeholders::_1));
  }

private:
  void status_callback(const std_msgs::msg::String::SharedPtr message)
  {
    RCLCPP_INFO(
      this->get_logger(),
      "Received: '%s'",
      message->data.c_str());
  }

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::spin(std::make_shared<StatusSubscriber>());

  rclcpp::shutdown();
  return 0;
}
