#include <chrono>
#include <functional>
#include <memory>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "robustlearn_interfaces/action/execute_system_check.hpp"

using namespace std::chrono_literals;

class SystemCheckActionServer : public rclcpp::Node
{
public:
  using ExecuteSystemCheck =
    robustlearn_interfaces::action::ExecuteSystemCheck;

  using GoalHandleSystemCheck =
    rclcpp_action::ServerGoalHandle<ExecuteSystemCheck>;

  SystemCheckActionServer()
  : Node("system_check_action_server")
  {
    action_server_ =
      rclcpp_action::create_server<ExecuteSystemCheck>(
      this,
      "execute_system_check",
      std::bind(
        &SystemCheckActionServer::handle_goal,
        this,
        std::placeholders::_1,
        std::placeholders::_2),
      std::bind(
        &SystemCheckActionServer::handle_cancel,
        this,
        std::placeholders::_1),
      std::bind(
        &SystemCheckActionServer::handle_accepted,
        this,
        std::placeholders::_1));

    RCLCPP_INFO(
      this->get_logger(),
      "System check action server ready");
  }

private:
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const ExecuteSystemCheck::Goal> goal)
  {
    (void)uuid;

    RCLCPP_INFO(
      this->get_logger(),
      "Received system check goal with %d steps",
      goal->total_steps);

    if (goal->total_steps <= 0) {
      RCLCPP_WARN(
        this->get_logger(),
        "Rejecting goal: total_steps must be greater than zero");

      return rclcpp_action::GoalResponse::REJECT;
    }

    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandleSystemCheck> goal_handle)
  {
    (void)goal_handle;

    RCLCPP_INFO(
      this->get_logger(),
      "Received request to cancel system check");

    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(
    const std::shared_ptr<GoalHandleSystemCheck> goal_handle)
  {
    std::thread{
      std::bind(
        &SystemCheckActionServer::execute,
        this,
        goal_handle)}
    .detach();
  }

  void execute(
    const std::shared_ptr<GoalHandleSystemCheck> goal_handle)
  {
    const auto goal = goal_handle->get_goal();

    auto feedback =
      std::make_shared<ExecuteSystemCheck::Feedback>();

    auto result =
      std::make_shared<ExecuteSystemCheck::Result>();

    for (int32_t step = 1; step <= goal->total_steps; ++step) {
      if (goal_handle->is_canceling()) {
        result->success = false;
        result->message = "System check canceled";

        goal_handle->canceled(result);

        RCLCPP_INFO(
          this->get_logger(),
          "System check canceled");

        return;
      }

      std::this_thread::sleep_for(500ms);

      if (goal_handle->is_canceling()) {
        result->success = false;
        result->message = "System check canceled";

        goal_handle->canceled(result);

        RCLCPP_INFO(
          this->get_logger(),
          "System check canceled");

        return;
      }

      feedback->completed_steps = step;
      feedback->progress =
        static_cast<float>(step) /
        static_cast<float>(goal->total_steps);

      goal_handle->publish_feedback(feedback);

      RCLCPP_INFO(
        this->get_logger(),
        "System check progress: %d/%d",
        step,
        goal->total_steps);
    }

    result->success = true;
    result->message = "System check completed";

    goal_handle->succeed(result);

    RCLCPP_INFO(
      this->get_logger(),
      "System check completed successfully");
  }

  rclcpp_action::Server<ExecuteSystemCheck>::SharedPtr
    action_server_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::spin(
    std::make_shared<SystemCheckActionServer>());

  rclcpp::shutdown();

  return 0;
}
