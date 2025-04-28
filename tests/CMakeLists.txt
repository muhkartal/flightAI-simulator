add_executable(test_advantage test_advantage.cpp)
target_link_libraries(test_advantage rldronesim_lib GTest::gtest GTest::gtest_main)

add_executable(test_ppo test_ppo.cpp)
target_link_libraries(test_ppo rldronesim_lib GTest::gtest GTest::gtest_main)

add_executable(test_grpc test_grpc.cpp)
target_link_libraries(test_grpc rldronesim_lib GTest::gtest GTest::gtest_main)

add_test(NAME test_advantage COMMAND test_advantage)
add_test(NAME test_ppo COMMAND test_ppo)
add_test(NAME test_grpc COMMAND test_grpc)

# Add coverage
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif()
