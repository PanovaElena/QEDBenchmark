cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)

option(USE_OMP ON)

project(QED_benchmark)

if (USE_OMP)
	add_definitions(-D__USE_OMP__)

	find_package(OpenMP)
	if (OPENMP_FOUND)
		set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	endif()
endif()

include_directories(
	hichi/Core/include/
	hichi/Particle/include/
	)

add_executable(QED_benchmark
	hichi/Particle/include/QED_AEG.h
	benchmark/main.cpp)

