include_guard(GLOBAL)

# Enforce minimum language standard as C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Prevent compiler-specific extensions from being used.
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable folder displays in development environments
set_property(GLOBAL PROPERTY USE_FOLDERS ON)