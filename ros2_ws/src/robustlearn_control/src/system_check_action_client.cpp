#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "robustlearn_interfaces/action/execute_system_check.hpp"

class SystemCheckActionClient : public rclcpp::Node
{
public:
  using ExecuteSystemCheck =
    robustlearn_interfaces::action::ExecuteSystemCheck;

  using GoalHandleSystemCheck =
    rclcpp_action::ClientGoalHandle<ExecuteSystemCheck>;

  SystemCheckActionClient(
    int32_t total_steps,
    int32_t cancel_after_step)
  : Node("system_check_action_client"),
    total_steps_(total_steps),
    cancel_after_step_(cancel_after_step),
    cancel_requested_(false)
  {
    action_client_ =
      rclcpp_action::create_client<ExecuteSystemCheck>(
      this,
      "execute_system_check");
  }

  void send_goal()
  {
    if (!action_client_->wait_for_action_server()) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Action server not available");

      rclcpp::shutdown();
      return;
    }

    ExecuteSystemCheck::Goal goal;
    goal.total_steps = total_steps_;

    RCLCPP_INFO(
      this->get_logger(),
      "Sending system check goal with %d steps",
      total_steps_);

    auto options =
      rclcpp_action::Client<ExecuteSystemCheck>::SendGoalOptions();

    options.goal_response_callback =
      std::bind(
      &SystemCheckActionClient::goal_response_callback,
      this,
      std::placeholders::_1);

    options.feedback_callback =
      std::bind(
      &SystemCheckActionClient::feedback_callback,
      this,
      std::placeholders::_1,
      std::placeholders::_2);

    options.result_callback =
      std::bind(
      &SystemCheckActionClient::result_callback,
      this,
      std::placeholders::_1);

    action_client_->async_send_goal(goal, options);
  }

private:
  void goal_response_callback(
    const GoalHandleSystemCheck::SharedPtr & goal_handle)
  {
    if (!goal_handle) {
      RCLCPP_WARN(
        this->get_logger(),
        "Goal was rejected");

      rclcpp::shutdown();
      return;
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Goal accepted");
  }

  void feedback_callback(
    GoalHandleSystemCheck::SharedPtr goal_handle,
    const std::shared_ptr<const ExecuteSystemCheck::Feedback>
    feedback)
  {
    RCLCPP_INFO(
      this->get_logger(),
      "Feedback: %d steps completed, progress %.2f",
      feedback->completed_steps,
      feedback->progress);

    if (
      cancel_after_step_ > 0 &&
      feedback->completed_steps >= cancel_after_step_ &&
      !cancel_requested_)
    {
      cancel_requested_ = true;

      RCLCPP_WARN(
        this->get_logger(),
        "Requesting cancellation after step %d",
        feedback->completed_steps);

      action_client_->async_cancel_goal(goal_handle);
    }
  }

  void result_callback(
    const GoalHandleSystemCheck::WrappedResult & result)
  {
    switch (result.code) {
      case rclcpp_action::ResultCode::SUCCEEDED:
        RCLCPP_INFO(
          this->get_logger(),
          "Result: success=%s, message='%s'",
          result.result->success ? "true" : "false",
          result.result->message.c_str());
        break;

      case rclcpp_action::ResultCode::ABORTED:
        RCLCPP_ERROR(
          this->get_logger(),
          "Goal was aborted");
        break;

      case rclcpp_action::ResultCode::CANCELED:
        RCLCPP_WARN(
          this->get_logger(),
          "Goal was canceled: '%s'",
          result.result->message.c_str());
        break;

      default:
        RCLCPP_ERROR(
          this->get_logger(),
          "Unknown action result");
        break;
    }

    rclcpp::shutdown();
  }

  int32_t total_steps_;
  int32_t cancel_after_step_;
  bool cancel_requested_;

  rclcpp_action::Client<ExecuteSystemCheck>::SharedPtr
    action_client_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  int32_t total_steps = 5;
  int32_t cancel_after_step = 0;

  if (argc > 1) {
    total_steps =
      static_cast<int32_t>(std::stoi(argv[1]));
  }

  if (argc > 2) {
    cancel_after_step =
      static_cast<int32_t>(std::stoi(argv[2]));
  }

  auto node =
    std::make_shared<SystemCheckActionClient>(
    total_steps,
    cancel_after_step);

  node->send_goal();

  rclcpp::spin(node);

  return 0;
}
