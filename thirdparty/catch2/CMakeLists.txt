file(GLOB_RECURSE INC_FILES "include/*")

# Add catch2 as an interface, as it is a header-only library
add_library(catch2 INTERFACE)
target_include_directories(catch2 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Enforce generation of "catch" as a target so that we get to see it in MSVS tree:
add_custom_target(catch SOURCES ${INC_FILES})