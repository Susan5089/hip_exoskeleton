cmake_minimum_required(VERSION 2.8.6)
project(mss)

#uncomment out the line below to switch to debug build or just runn "cmake -DCMAKE_BUILD_TYPE=Debug .."
#set(CMAKE_BUILD_TYPE Debug CACHE STRING "Debug|Release|RelWithDebInfo|MinSizeRel" FORCE)

#comment this line if you want to use boost version other than 1.66
set(BOOST_166 on)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)
#dubug mode
set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g")

add_compile_options(-fPIC)
add_compile_options(-std=gnu++11)
add_compile_options(-Wdeprecated-declarations)

add_compile_options(-DHAVE_CSTDDEF)
add_definitions(-DMASS_ROOT_DIR="${CMAKE_HOME_DIRECTORY}")
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake")

add_subdirectory( core )
add_subdirectory( render )
add_subdirectory( python )

if(BOOST_166)
  find_package(Boost 1.66.0 COMPONENTS filesystem python3 numpy3)
  #message("Boost_LIBRARIES 166: ${Boost_LIBRARIES}")
else(BOOST_166)
  find_package(Boost 1.72.0 COMPONENTS filesystem regex system python36 numpy36)
  #message("Boost_LIBRARIES 172: ${Boost_LIBRARIES}")
endif(BOOST_166)

if(Boost_FOUND)
  message("found Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
  message("found Boost_LIBRARIES: ${Boost_LIBRARIES}")
endif()


find_package(DART REQUIRED COMPONENTS utils-urdf gui collision-bullet CONFIG)
find_package(PythonLibs REQUIRED)

include_directories(${DART_INCLUDE_DIRS})
include_directories("${PROJECT_SOURCE_DIR}/core")
include_directories("${PROJECT_SOURCE_DIR}/python")
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${DART_INCLUDE_DIRS})

#add_executable(Test test.cpp)
#target_link_libraries (Test mss ${DART_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} pymss)
#target_link_libraries (Test pymss)
#target_link_libraries(Test ${DART_LIBRARIES})
