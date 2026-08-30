#include <chrono>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "robustlearn_interfaces/srv/set_system_mode.hpp"

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("system_mode_client");

  using SetSystemMode = robustlearn_interfaces::srv::SetSystemMode;

  auto client = node->create_client<SetSystemMode>("set_system_mode");

  while (!client->wait_for_service(1s)) {
    if (!rclcpp::ok()) {
      RCLCPP_ERROR(
        node->get_logger(),
        "Interrupted while waiting for the service");
      rclcpp::shutdown();
      return 1;
    }

    RCLCPP_INFO(
      node->get_logger(),
      "Waiting for /set_system_mode service...");
  }

  auto request = std::make_shared<SetSystemMode::Request>();

  if (argc > 1) {
    request->mode = argv[1];
  } else {
    request->mode = "READY";
  }

  RCLCPP_INFO(
    node->get_logger(),
    "Requesting system mode '%s'",
    request->mode.c_str());

  auto future = client->async_send_request(request);

  if (
    rclcpp::spin_until_future_complete(node, future) ==
    rclcpp::FutureReturnCode::SUCCESS)
  {
    const auto response = future.get();

    RCLCPP_INFO(
      node->get_logger(),
      "success=%s, message='%s'",
      response->success ? "true" : "false",
      response->message.c_str());
  } else {
    RCLCPP_ERROR(
      node->get_logger(),
      "Service call failed");

    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
