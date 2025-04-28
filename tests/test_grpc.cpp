#include <gtest/gtest.h>
#include "airsim_bridge.grpc.pb.h"
#include "grpc_client.h"
#include <grpcpp/grpcpp.h>
#include <thread>
#include <memory>

using airsim_bridge::DroneControl;
using airsim_bridge::DroneAction;
using airsim_bridge::DroneObservation;
using airsim_bridge::ResetRequest;
using airsim_bridge::Vector3;
using airsim_bridge::Quaternion;

class MockDroneControlService : public DroneControl::Service {
public:
    grpc::Status ControlLoop(
        grpc::ServerReaderWriter<DroneObservation, DroneAction>* stream) override {

        // Send initial observation
        DroneObservation initial_obs = create_observation();
        stream->Write(initial_obs);

        // Echo back actions as observations
        DroneAction action;
        while (stream->Read(&action)) {
            DroneObservation obs = create_observation();
            obs.mutable_position()->set_x(action.roll());
            obs.mutable_position()->set_y(action.pitch());
            obs.mutable_position()->set_z(action.throttle());
            obs.mutable_angular_velocity()->set_z(action.yaw_rate());

            if (!stream->Write(obs)) {
                return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Failed to write observation");
            }
        }

        return grpc::Status::OK;
    }

    grpc::Status ResetEnvironment(
        grpc::ServerContext* context,
        const ResetRequest* request,
        DroneObservation* response) override {

        *response = create_observation();

        if (request->random_start()) {
            response->mutable_position()->set_x((float)rand() / RAND_MAX * 10.0f - 5.0f);
            response->mutable_position()->set_y((float)rand() / RAND_MAX * 10.0f - 5.0f);
            response->mutable_position()->set_z((float)rand() / RAND_MAX * -10.0f);
        } else if (request->has_initial_position()) {
            response->mutable_position()->set_x(request->initial_position().x());
            response->mutable_position()->set_y(request->initial_position().y());
            response->mutable_position()->set_z(request->initial_position().z());
        }

        return grpc::Status::OK;
    }

private:
    DroneObservation create_observation() {
        DroneObservation obs;

        obs.mutable_position()->set_x(0.0f);
        obs.mutable_position()->set_y(0.0f);
        obs.mutable_position()->set_z(-2.0f);

        obs.mutable_linear_velocity()->set_x(0.0f);
        obs.mutable_linear_velocity()->set_y(0.0f);
        obs.mutable_linear_velocity()->set_z(0.0f);

        obs.mutable_angular_velocity()->set_x(0.0f);
        obs.mutable_angular_velocity()->set_y(0.0f);
        obs.mutable_angular_velocity()->set_z(0.0f);

        obs.mutable_orientation()->set_w(1.0f);
        obs.mutable_orientation()->set_x(0.0f);
        obs.mutable_orientation()->set_y(0.0f);
        obs.mutable_orientation()->set_z(0.0f);

        obs.set_collision_info(0.0f);
        obs.set_reward(0.0f);
        obs.set_done(false);
        obs.set_timestamp_ns(0);

        return obs;
    }
};

class GrpcTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Start server
        server_address_ = "localhost:50052";

        grpc::ServerBuilder builder;
        builder.AddListeningPort(server_address_, grpc::InsecureServerCredentials());
        builder.RegisterService(&service_);

        server_ = builder.BuildAndStart();
        ASSERT_NE(server_, nullptr);

        // Give server time to start
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    void TearDown() override {
        server_->Shutdown();
    }

    std::string server_address_;
    MockDroneControlService service_;
    std::unique_ptr<grpc::Server> server_;
};

TEST_F(GrpcTest, ConnectAndReset) {
    // Create client
    GrpcClient client(server_address_);

    // Connect to server
    EXPECT_TRUE(client.connect());

    // Reset environment
    auto obs = client.reset(true);

    // Check we got a valid observation
    EXPECT_EQ(obs.position().z(), -2.0f);

    // Disconnect
    client.disconnect();
}

TEST_F(GrpcTest, SendAction) {
    // Create client
    GrpcClient client(server_address_);

    // Connect to server
    EXPECT_TRUE(client.connect());

    // Reset environment
    client.reset(false);

    // Send action
    DroneAction action;
    action.set_roll(0.5f);
    action.set_pitch(-0.3f);
    action.set_yaw_rate(0.1f);
    action.set_throttle(0.7f);
    action.set_timestamp_ns(1234567890);

    auto obs = client.step(action);

    // Check that action values were echoed back in the observation
    EXPECT_FLOAT_EQ(obs.position().x(), 0.5f);
    EXPECT_FLOAT_EQ(obs.position().y(), -0.3f);
    EXPECT_FLOAT_EQ(obs.position().z(), 0.7f);
    EXPECT_FLOAT_EQ(obs.angular_velocity().z(), 0.1f);

    // Disconnect
    client.disconnect();
}
