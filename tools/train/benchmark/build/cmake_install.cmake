# Install script for directory: /Users/cdq/Desktop/MNN

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/Users/cdq/Desktop/android-ndk-r21b/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android-objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/MNN" TYPE FILE FILES
    "/Users/cdq/Desktop/MNN/include/MNN/MNNDefine.h"
    "/Users/cdq/Desktop/MNN/include/MNN/Interpreter.hpp"
    "/Users/cdq/Desktop/MNN/include/MNN/HalideRuntime.h"
    "/Users/cdq/Desktop/MNN/include/MNN/Tensor.hpp"
    "/Users/cdq/Desktop/MNN/include/MNN/ErrorCode.hpp"
    "/Users/cdq/Desktop/MNN/include/MNN/ImageProcess.hpp"
    "/Users/cdq/Desktop/MNN/include/MNN/Matrix.h"
    "/Users/cdq/Desktop/MNN/include/MNN/Rect.h"
    "/Users/cdq/Desktop/MNN/include/MNN/MNNForwardType.h"
    "/Users/cdq/Desktop/MNN/include/MNN/AutoTime.hpp"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/MNN/expr" TYPE FILE FILES
    "/Users/cdq/Desktop/MNN/include/MNN/expr/Expr.hpp"
    "/Users/cdq/Desktop/MNN/include/MNN/expr/ExprCreator.hpp"
    "/Users/cdq/Desktop/MNN/include/MNN/expr/MathOp.hpp"
    "/Users/cdq/Desktop/MNN/include/MNN/expr/NeuralNetWorkOp.hpp"
    "/Users/cdq/Desktop/MNN/include/MNN/expr/Optimizer.hpp"
    "/Users/cdq/Desktop/MNN/include/MNN/expr/Executor.hpp"
    "/Users/cdq/Desktop/MNN/include/MNN/expr/NN.hpp"
    "/Users/cdq/Desktop/MNN/include/MNN/expr/Module.hpp"
    "/Users/cdq/Desktop/MNN/include/MNN/expr/NeuralNetWorkOp.hpp"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/Users/cdq/Desktop/MNN/tools/train/benchmark/build/libMNN.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Users/cdq/Desktop/android-ndk-r21b/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/cdq/Desktop/MNN/tools/train/benchmark/build/express/cmake_install.cmake")
  include("/Users/cdq/Desktop/MNN/tools/train/benchmark/build/tools/train/cmake_install.cmake")
  include("/Users/cdq/Desktop/MNN/tools/train/benchmark/build/tools/converter/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/Users/cdq/Desktop/MNN/tools/train/benchmark/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
