#include "agent.h"
#include "grpc_server.h"
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include <thread>
#include <csignal>

std::unique_ptr<GrpcServer> g_server;
bool g_running = true;

void signal_handler(int signal) {
    spdlog::info("Received signal {}, shutting down...", signal);
    g_running = false;

    if (g_server) {
        g_server->stop();
    }
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    spdlog::set_level(spdlog::level::info);
    spdlog::info("RL-DroneSim starting up");

    CLI::App app{"RL-DroneSim - Reinforcement Learning for Drone Simulation"};

    std::string config_path = "configs/training_config.yaml";
    app.add_option("--config", config_path, "Path to configuration file");

    bool server_mode = false;
    app.add_flag("--server", server_mode, "Run in server mode (AirSim bridge only)");

    int server_port = 50051;
    app.add_option("--port", server_port, "Server port");

    bool verbose = false;
    app.add_flag("-v,--verbose", verbose, "Enable verbose logging");

    std::string checkpoint_path;

    CLI::App* train_cmd = app.add_subcommand("train", "Train the agent");
    int train_epochs = 100;
    train_cmd->add_option("--epochs", train_epochs, "Number of epochs to train");

    CLI::App* eval_cmd = app.add_subcommand("eval", "Evaluate the agent");
    int eval_episodes = 10;
    bool record = false;
    eval_cmd->add_option("--episodes", eval_episodes, "Number of episodes to evaluate");
    eval_cmd->add_flag("--record", record, "Record evaluation to video");
    eval_cmd->add_option("--checkpoint", checkpoint_path, "Path to checkpoint file");

    CLI::App* play_cmd = app.add_subcommand("play", "Play with manual control");

    app.require_subcommand(1);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    if (verbose) {
        spdlog::set_level(spdlog::level::debug);
    }

    if (server_mode) {
        // Run only the AirSim bridge server
        std::string server_address = "0.0.0.0:" + std::to_string(server_port);
        g_server = std::make_unique<GrpcServer>(server_address);
        g_server->start();

        while (g_running) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        return 0;
    }

    try {
        if (train_cmd->parsed()) {
            // Training mode
            Agent agent(config_path);
            agent.train(train_epochs);

        } else if (eval_cmd->parsed()) {
            // Evaluation mode
            Agent agent(config_path);

            if (!checkpoint_path.empty()) {
                agent.load_checkpoint(checkpoint_path);
            }

            agent.evaluate(eval_episodes, record);

        } else if (play_cmd->parsed()) {
            // Interactive play mode
            Agent agent(config_path);

            if (!checkpoint_path.empty()) {
                agent.load_checkpoint(checkpoint_path);
            }

            agent.play_interactive();
        }
    } catch (const std::exception& e) {
        spdlog::error("Error: {}", e.what());
        return 1;
    }

    spdlog::info("RL-DroneSim shutting down");
    return 0;
}
