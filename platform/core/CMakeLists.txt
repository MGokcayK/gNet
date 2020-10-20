file(GLOB_RECURSE INC_FILES "include/*.hpp")
file(GLOB_RECURSE SRC_FILES "source/*.cpp")

add_library(core STATIC ${INC_FILES} ${SRC_FILES})

target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(core PUBLIC eigen)