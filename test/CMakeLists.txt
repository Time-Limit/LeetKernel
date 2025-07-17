# test/CMakeLists.txt

add_library(kahan OBJECT kahan.cu)

add_executable(test test.cc $<TARGET_OBJECTS:kahan>)

target_link_libraries(test PRIVATE llmmm)
