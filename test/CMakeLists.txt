# test/CMakeLists.txt

add_library(kahan OBJECT kahan.cu)

add_executable(test_verify test_verify.cc $<TARGET_OBJECTS:kahan>)

target_link_libraries(test_verify PRIVATE llmmm)

add_executable(test_tune test_tune.cc)

target_link_libraries(test_tune PRIVATE llmmm)
