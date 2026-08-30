#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "robustlearn_interfaces/srv/set_system_mode.hpp"

class SystemModeServer : public rclcpp::Node
{
public:
  using SetSystemMode = robustlearn_interfaces::srv::SetSystemMode;

  SystemModeServer()
  : Node("system_mode_server"), current_mode_("IDLE")
  {
    service_ = this->create_service<SetSystemMode>(
      "set_system_mode",
      std::bind(
        &SystemModeServer::handle_request,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    RCLCPP_INFO(
      this->get_logger(),
      "System mode service ready");
  }

private:
  void handle_request(
    const std::shared_ptr<SetSystemMode::Request> request,
    std::shared_ptr<SetSystemMode::Response> response)
  {
    if (request->mode.empty()) {
      response->success = false;
      response->message = "Mode must not be empty";
      return;
    }

    current_mode_ = request->mode;

    response->success = true;
    response->message = "System mode set to " + current_mode_;

    RCLCPP_INFO(
      this->get_logger(),
      "System mode changed to '%s'",
      current_mode_.c_str());
  }

  std::string current_mode_;
  rclcpp::Service<SetSystemMode>::SharedPtr service_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SystemModeServer>());
  rclcpp::shutdown();
  return 0;
}
