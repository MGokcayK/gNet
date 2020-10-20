file(GLOB_RECURSE INC_FILES "include/*")

# Eigen is a header-only library, therefore set it up as an interface library.
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# If we don't add the line below, we can't see the project in MSVS:
add_custom_target(eigen3 SOURCES ${INC_FILES})