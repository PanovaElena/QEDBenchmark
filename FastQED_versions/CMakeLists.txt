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

if (UNIX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=max")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qopt-report=max")
endif()

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin/)

include_directories(
	${CMAKE_SOURCE_DIR}/../hichi/Core/include/
	${CMAKE_SOURCE_DIR}/../hichi/Particle/include/
	)

# baseline
add_executable(QED_benchmark_v0
	${CMAKE_SOURCE_DIR}/par_v0_critical/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v0_critical/compton.h
	${CMAKE_SOURCE_DIR}/par_v0_critical/Fast_QED.h
	${CMAKE_SOURCE_DIR}/par_v0_critical/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v0_critical/Particle.h
	${CMAKE_SOURCE_DIR}/src/macros.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v0 PUBLIC __VER_QED_BENCHMARK__="v0")
target_include_directories(QED_benchmark_v0 PUBLIC ${CMAKE_SOURCE_DIR}/par_v0_critical/)

# non-crtitical
add_executable(QED_benchmark_v1
	${CMAKE_SOURCE_DIR}/par_v1_nocritical/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v1_nocritical/compton.h
	${CMAKE_SOURCE_DIR}/par_v1_nocritical/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v1_nocritical/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v1_nocritical/Particle.h
	${CMAKE_SOURCE_DIR}/src/macros.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v1 PUBLIC __VER_QED_BENCHMARK__="v1")
target_include_directories(QED_benchmark_v1 PUBLIC ${CMAKE_SOURCE_DIR}/par_v1_nocritical/)

# optimized vectors
add_executable(QED_benchmark_v2_1
	${CMAKE_SOURCE_DIR}/par_v2_1_reserve/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v2_1_reserve/compton.h
	${CMAKE_SOURCE_DIR}/par_v2_1_reserve/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v2_1_reserve/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v2_1_reserve/Particle.h
	${CMAKE_SOURCE_DIR}/src/macros.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v2_1 PUBLIC __VER_QED_BENCHMARK__="v2_1")
target_include_directories(QED_benchmark_v2_1 PUBLIC ${CMAKE_SOURCE_DIR}/par_v2_1_reserve/)

add_executable(QED_benchmark_v2_2
	${CMAKE_SOURCE_DIR}/par_v2_2_newvec/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v2_2_newvec/compton.h
	${CMAKE_SOURCE_DIR}/par_v2_2_newvec/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v2_2_newvec/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v2_2_newvec/Particle.h
	${CMAKE_SOURCE_DIR}/src/macros.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v2_2 PUBLIC __VER_QED_BENCHMARK__="v2_2")
target_include_directories(QED_benchmark_v2_2 PUBLIC ${CMAKE_SOURCE_DIR}/par_v2_2_newvec/)

add_executable(QED_benchmark_v2_3
	${CMAKE_SOURCE_DIR}/par_v2_3_avoid_of_small_vectors/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v2_3_avoid_of_small_vectors/compton.h
	${CMAKE_SOURCE_DIR}/par_v2_3_avoid_of_small_vectors/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v2_3_avoid_of_small_vectors/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v2_3_avoid_of_small_vectors/Particle.h
	${CMAKE_SOURCE_DIR}/src/macros.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v2_3 PUBLIC __VER_QED_BENCHMARK__="v2_3")
target_include_directories(QED_benchmark_v2_3 PUBLIC ${CMAKE_SOURCE_DIR}/par_v2_3_avoid_of_small_vectors/)

add_executable(QED_benchmark_v2_4
	${CMAKE_SOURCE_DIR}/par_v2_4_myvec/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v2_4_myvec/compton.h
	${CMAKE_SOURCE_DIR}/par_v2_4_myvec/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v2_4_myvec/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v2_4_myvec/Particle.h
	${CMAKE_SOURCE_DIR}/src/macros.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v2_4 PUBLIC __VER_QED_BENCHMARK__="v2_4")
target_include_directories(QED_benchmark_v2_4 PUBLIC ${CMAKE_SOURCE_DIR}/par_v2_4_myvec/)

# vectorization
add_executable(QED_benchmark_v3_1
	${CMAKE_SOURCE_DIR}/par_v3_1_chunks/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v3_1_chunks/compton.h
	${CMAKE_SOURCE_DIR}/par_v3_1_chunks/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v3_1_chunks/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v3_1_chunks/Particle.h
	${CMAKE_SOURCE_DIR}/src/macros.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v3_1 PUBLIC __VER_QED_BENCHMARK__="v3_1")
target_include_directories(QED_benchmark_v3_1 PUBLIC ${CMAKE_SOURCE_DIR}/par_v3_1_chunks/)

add_executable(QED_benchmark_v3_2
	${CMAKE_SOURCE_DIR}/par_v3_2_try_vectorize_one_loop/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v3_2_try_vectorize_one_loop/compton.h
	${CMAKE_SOURCE_DIR}/par_v3_2_try_vectorize_one_loop/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v3_2_try_vectorize_one_loop/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v3_2_try_vectorize_one_loop/Particle.h
	${CMAKE_SOURCE_DIR}/src/macros.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v3_2 PUBLIC __VER_QED_BENCHMARK__="v3_2")
target_include_directories(QED_benchmark_v3_2 PUBLIC ${CMAKE_SOURCE_DIR}/par_v3_2_try_vectorize_one_loop/)

add_executable(QED_benchmark_v3_3
	${CMAKE_SOURCE_DIR}/par_v3_3_try_vectorize_one_loop_pragma_simd/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v3_3_try_vectorize_one_loop_pragma_simd/compton.h
	${CMAKE_SOURCE_DIR}/par_v3_3_try_vectorize_one_loop_pragma_simd/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v3_3_try_vectorize_one_loop_pragma_simd/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v3_3_try_vectorize_one_loop_pragma_simd/Particle.h
	${CMAKE_SOURCE_DIR}/src/macros.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v3_3 PUBLIC __VER_QED_BENCHMARK__="v3_3")
target_include_directories(QED_benchmark_v3_3 PUBLIC ${CMAKE_SOURCE_DIR}/par_v3_3_try_vectorize_one_loop_pragma_simd/)

add_executable(QED_benchmark_v3_4
	${CMAKE_SOURCE_DIR}/par_v3_4_try_vectorize_many_small_loops/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v3_4_try_vectorize_many_small_loops/compton.h
	${CMAKE_SOURCE_DIR}/par_v3_4_try_vectorize_many_small_loops/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v3_4_try_vectorize_many_small_loops/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v3_4_try_vectorize_many_small_loops/Particle.h
	${CMAKE_SOURCE_DIR}/src/macros.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v3_4 PUBLIC __VER_QED_BENCHMARK__="v3_4")
target_include_directories(QED_benchmark_v3_4 PUBLIC ${CMAKE_SOURCE_DIR}/par_v3_4_try_vectorize_many_small_loops/)

add_executable(QED_benchmark_v3_5
	${CMAKE_SOURCE_DIR}/par_v3_5_vectorized_combined_loops/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v3_5_vectorized_combined_loops/compton.h
	${CMAKE_SOURCE_DIR}/par_v3_5_vectorized_combined_loops/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v3_5_vectorized_combined_loops/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v3_5_vectorized_combined_loops/Particle.h
	${CMAKE_SOURCE_DIR}/src/macros.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v3_5 PUBLIC __VER_QED_BENCHMARK__="v3_5")
target_include_directories(QED_benchmark_v3_5 PUBLIC ${CMAKE_SOURCE_DIR}/par_v3_5_vectorized_combined_loops/)

