cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)

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

if (NOT ITER)
    set(ITER 61)
endif()

if (NOT ITER_DUMP)
    set(ITER_DUMP 10)
endif()

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin/)

include_directories(
	${CMAKE_SOURCE_DIR}/../hichi/Core/include/
	${CMAKE_SOURCE_DIR}/../hichi/Particle/include/
	)

# baseline
add_executable(QED_benchmark_v0
	${CMAKE_SOURCE_DIR}/par_v0_critical/macros.h
	${CMAKE_SOURCE_DIR}/par_v0_critical/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v0_critical/compton.h
	${CMAKE_SOURCE_DIR}/par_v0_critical/Fast_QED.h
	${CMAKE_SOURCE_DIR}/par_v0_critical/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v0_critical/Particle.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v0 PUBLIC __VER_QED_BENCHMARK__="v0" PUBLIC __ITER_NUM__=${ITER} PUBLIC __ITER_DUMP__=${ITER_DUMP})
target_include_directories(QED_benchmark_v0 PUBLIC ${CMAKE_SOURCE_DIR}/par_v0_critical/)

# non-crtitical
add_executable(QED_benchmark_v1
	${CMAKE_SOURCE_DIR}/par_v1_nocritical/macros.h
	${CMAKE_SOURCE_DIR}/par_v1_nocritical/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v1_nocritical/compton.h
	${CMAKE_SOURCE_DIR}/par_v1_nocritical/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v1_nocritical/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v1_nocritical/Particle.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v1 PUBLIC __VER_QED_BENCHMARK__="v1" PUBLIC __ITER_NUM__=${ITER} PUBLIC __ITER_DUMP__=${ITER_DUMP})
target_include_directories(QED_benchmark_v1 PUBLIC ${CMAKE_SOURCE_DIR}/par_v1_nocritical/)

# optimized vectors
add_executable(QED_benchmark_v2_1
	${CMAKE_SOURCE_DIR}/par_v2_1_reserve/macros.h
	${CMAKE_SOURCE_DIR}/par_v2_1_reserve/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v2_1_reserve/compton.h
	${CMAKE_SOURCE_DIR}/par_v2_1_reserve/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v2_1_reserve/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v2_1_reserve/Particle.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v2_1 PUBLIC __VER_QED_BENCHMARK__="v2_1" PUBLIC __ITER_NUM__=${ITER} PUBLIC __ITER_DUMP__=${ITER_DUMP})
target_include_directories(QED_benchmark_v2_1 PUBLIC ${CMAKE_SOURCE_DIR}/par_v2_1_reserve/)

add_executable(QED_benchmark_v2_2
	${CMAKE_SOURCE_DIR}/par_v2_2_newvec/macros.h
	${CMAKE_SOURCE_DIR}/par_v2_2_newvec/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v2_2_newvec/compton.h
	${CMAKE_SOURCE_DIR}/par_v2_2_newvec/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v2_2_newvec/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v2_2_newvec/Particle.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v2_2 PUBLIC __VER_QED_BENCHMARK__="v2_2" PUBLIC __ITER_NUM__=${ITER} PUBLIC __ITER_DUMP__=${ITER_DUMP})
target_include_directories(QED_benchmark_v2_2 PUBLIC ${CMAKE_SOURCE_DIR}/par_v2_2_newvec/)

add_executable(QED_benchmark_v2_3
	${CMAKE_SOURCE_DIR}/par_v2_3_avoid_of_small_vectors/macros.h
	${CMAKE_SOURCE_DIR}/par_v2_3_avoid_of_small_vectors/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v2_3_avoid_of_small_vectors/compton.h
	${CMAKE_SOURCE_DIR}/par_v2_3_avoid_of_small_vectors/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v2_3_avoid_of_small_vectors/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v2_3_avoid_of_small_vectors/Particle.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v2_3 PUBLIC __VER_QED_BENCHMARK__="v2_3" PUBLIC __ITER_NUM__=${ITER} PUBLIC __ITER_DUMP__=${ITER_DUMP})
target_include_directories(QED_benchmark_v2_3 PUBLIC ${CMAKE_SOURCE_DIR}/par_v2_3_avoid_of_small_vectors/)

add_executable(QED_benchmark_v2_4
	${CMAKE_SOURCE_DIR}/par_v2_4_myvec/macros.h
	${CMAKE_SOURCE_DIR}/par_v2_4_myvec/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v2_4_myvec/compton.h
	${CMAKE_SOURCE_DIR}/par_v2_4_myvec/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v2_4_myvec/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v2_4_myvec/Particle.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v2_4 PUBLIC __VER_QED_BENCHMARK__="v2_4" PUBLIC __ITER_NUM__=${ITER} PUBLIC __ITER_DUMP__=${ITER_DUMP})
target_include_directories(QED_benchmark_v2_4 PUBLIC ${CMAKE_SOURCE_DIR}/par_v2_4_myvec/)

add_executable(QED_benchmark_v2_5
	${CMAKE_SOURCE_DIR}/par_v2_5_my_ideal_vec/macros.h
	${CMAKE_SOURCE_DIR}/par_v2_5_my_ideal_vec/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v2_5_my_ideal_vec/compton.h
	${CMAKE_SOURCE_DIR}/par_v2_5_my_ideal_vec/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v2_5_my_ideal_vec/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v2_5_my_ideal_vec/Particle.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v2_5 PUBLIC __VER_QED_BENCHMARK__="v2_5" PUBLIC __ITER_NUM__=${ITER} PUBLIC __ITER_DUMP__=${ITER_DUMP})
target_include_directories(QED_benchmark_v2_5 PUBLIC ${CMAKE_SOURCE_DIR}/par_v2_5_my_ideal_vec/)

add_executable(QED_benchmark_v2_6
	${CMAKE_SOURCE_DIR}/par_v2_6_copying_particles/macros.h
	${CMAKE_SOURCE_DIR}/par_v2_6_copying_particles/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v2_6_copying_particles/compton.h
	${CMAKE_SOURCE_DIR}/par_v2_6_copying_particles/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v2_6_copying_particles/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v2_6_copying_particles/Particle.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v2_6 PUBLIC __VER_QED_BENCHMARK__="v2_6" PUBLIC __ITER_NUM__=${ITER} PUBLIC __ITER_DUMP__=${ITER_DUMP})
target_include_directories(QED_benchmark_v2_6 PUBLIC ${CMAKE_SOURCE_DIR}/par_v2_6_copying_particles/)

# vectorization
add_executable(QED_benchmark_v3_1
	${CMAKE_SOURCE_DIR}/par_v3_1_chunks/macros.h
	${CMAKE_SOURCE_DIR}/par_v3_1_chunks/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v3_1_chunks/compton.h
	${CMAKE_SOURCE_DIR}/par_v3_1_chunks/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v3_1_chunks/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v3_1_chunks/Particle.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v3_1 PUBLIC __VER_QED_BENCHMARK__="v3_1" PUBLIC __ITER_NUM__=${ITER} PUBLIC __ITER_DUMP__=${ITER_DUMP})
target_include_directories(QED_benchmark_v3_1 PUBLIC ${CMAKE_SOURCE_DIR}/par_v3_1_chunks/)

add_executable(QED_benchmark_v3_2
	${CMAKE_SOURCE_DIR}/par_v3_2_try_vectorize_one_loop/macros.h
	${CMAKE_SOURCE_DIR}/par_v3_2_try_vectorize_one_loop/breit_wheeler.h
	${CMAKE_SOURCE_DIR}/par_v3_2_try_vectorize_one_loop/compton.h
	${CMAKE_SOURCE_DIR}/par_v3_2_try_vectorize_one_loop/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v3_2_try_vectorize_one_loop/Vectors.h
	${CMAKE_SOURCE_DIR}/par_v3_2_try_vectorize_one_loop/Particle.h
	${CMAKE_SOURCE_DIR}/src/main.cpp
	)
target_compile_definitions(QED_benchmark_v3_2 PUBLIC __VER_QED_BENCHMARK__="v3_2" PUBLIC __ITER_NUM__=${ITER} PUBLIC __ITER_DUMP__=${ITER_DUMP})
target_include_directories(QED_benchmark_v3_2 PUBLIC ${CMAKE_SOURCE_DIR}/par_v3_2_try_vectorize_one_loop/)

set(CHUNK_SIZES 8 16 32 64 128 256 512)
foreach(CHUNK IN LISTS CHUNK_SIZES)
    add_executable(QED_benchmark_v3_4_${CHUNK}
        ${CMAKE_SOURCE_DIR}/par_v3_4_try_vectorize_many_small_loops/macros.h
        ${CMAKE_SOURCE_DIR}/par_v3_4_try_vectorize_many_small_loops/breit_wheeler.h
        ${CMAKE_SOURCE_DIR}/par_v3_4_try_vectorize_many_small_loops/compton.h
        ${CMAKE_SOURCE_DIR}/par_v3_4_try_vectorize_many_small_loops/Fast_QED.h
        ${CMAKE_SOURCE_DIR}/par_v3_4_try_vectorize_many_small_loops/Vectors.h
        ${CMAKE_SOURCE_DIR}/par_v3_4_try_vectorize_many_small_loops/Particle.h
        ${CMAKE_SOURCE_DIR}/src/main.cpp
        )
    target_compile_definitions(QED_benchmark_v3_4_${CHUNK} PUBLIC __VER_QED_BENCHMARK__="v3_4_${CHUNK}" PUBLIC __CHUNK_SIZE__=${CHUNK} PUBLIC __ITER_NUM__=${ITER} PUBLIC __ITER_DUMP__=${ITER_DUMP})
    target_include_directories(QED_benchmark_v3_4_${CHUNK} PUBLIC ${CMAKE_SOURCE_DIR}/par_v3_4_try_vectorize_many_small_loops/)
    
    add_executable(QED_benchmark_v3_5_${CHUNK}
        ${CMAKE_SOURCE_DIR}/par_v3_5_vectorized_combined_loops/macros.h
        ${CMAKE_SOURCE_DIR}/par_v3_5_vectorized_combined_loops/breit_wheeler.h
        ${CMAKE_SOURCE_DIR}/par_v3_5_vectorized_combined_loops/compton.h
        ${CMAKE_SOURCE_DIR}/par_v3_5_vectorized_combined_loops/Fast_QED.h
        ${CMAKE_SOURCE_DIR}/par_v3_5_vectorized_combined_loops/Vectors.h
        ${CMAKE_SOURCE_DIR}/par_v3_5_vectorized_combined_loops/Particle.h
        ${CMAKE_SOURCE_DIR}/src/main.cpp
        )
    target_compile_definitions(QED_benchmark_v3_5_${CHUNK} PUBLIC __VER_QED_BENCHMARK__="v3_5_${CHUNK}" PUBLIC __CHUNK_SIZE__=${CHUNK} PUBLIC __ITER_NUM__=${ITER} PUBLIC __ITER_DUMP__=${ITER_DUMP})
    target_include_directories(QED_benchmark_v3_5_${CHUNK} PUBLIC ${CMAKE_SOURCE_DIR}/par_v3_5_vectorized_combined_loops/)
endforeach()

add_executable(QED_benchmark_v4
    ${CMAKE_SOURCE_DIR}/par_v4_vectorization_without_vector/macros.h
    ${CMAKE_SOURCE_DIR}/par_v4_vectorization_without_vector/breit_wheeler.h
    ${CMAKE_SOURCE_DIR}/par_v4_vectorization_without_vector/compton.h
    ${CMAKE_SOURCE_DIR}/par_v4_vectorization_without_vector/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v4_vectorization_without_vector/Vectors.h
    ${CMAKE_SOURCE_DIR}/par_v4_vectorization_without_vector/Particle.h
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    )
target_compile_definitions(QED_benchmark_v4 PUBLIC __VER_QED_BENCHMARK__="v4" PUBLIC __CHUNK_SIZE__=128 PUBLIC __ITER_NUM__=${ITER} PUBLIC __ITER_DUMP__=${ITER_DUMP})
target_include_directories(QED_benchmark_v4 PUBLIC ${CMAKE_SOURCE_DIR}/par_v4_vectorization_without_vector/)

add_executable(QED_benchmark_v5
    ${CMAKE_SOURCE_DIR}/par_v5_no_vectorization/macros.h
    ${CMAKE_SOURCE_DIR}/par_v5_no_vectorization/breit_wheeler.h
    ${CMAKE_SOURCE_DIR}/par_v5_no_vectorization/compton.h
    ${CMAKE_SOURCE_DIR}/par_v5_no_vectorization/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v5_no_vectorization/Vectors.h
    ${CMAKE_SOURCE_DIR}/par_v5_no_vectorization/Particle.h
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    )
target_compile_definitions(QED_benchmark_v5 PUBLIC __VER_QED_BENCHMARK__="v5" PUBLIC __CHUNK_SIZE__=128 PUBLIC __ITER_NUM__=${ITER} PUBLIC __ITER_DUMP__=${ITER_DUMP})
target_include_directories(QED_benchmark_v5 PUBLIC ${CMAKE_SOURCE_DIR}/par_v5_no_vectorization/)

add_executable(QED_benchmark_v6
    ${CMAKE_SOURCE_DIR}/par_v6_adapted_for_integration/macros.h
    ${CMAKE_SOURCE_DIR}/par_v6_adapted_for_integration/breit_wheeler.h
    ${CMAKE_SOURCE_DIR}/par_v6_adapted_for_integration/compton.h
    ${CMAKE_SOURCE_DIR}/par_v6_adapted_for_integration/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v6_adapted_for_integration/Vectors.h
    ${CMAKE_SOURCE_DIR}/par_v6_adapted_for_integration/Particle.h
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    )
target_compile_definitions(QED_benchmark_v6 PUBLIC __VER_QED_BENCHMARK__="v6" PUBLIC __CHUNK_SIZE__=128 PUBLIC __ITER_NUM__=${ITER} PUBLIC __ITER_DUMP__=${ITER_DUMP})
target_include_directories(QED_benchmark_v6 PUBLIC ${CMAKE_SOURCE_DIR}/par_v6_adapted_for_integration/)

add_executable(QED_benchmark_v7
    ${CMAKE_SOURCE_DIR}/par_v7_adapted_for_picador/macros.h
    ${CMAKE_SOURCE_DIR}/par_v7_adapted_for_picador/breit_wheeler.h
    ${CMAKE_SOURCE_DIR}/par_v7_adapted_for_picador/compton.h
    ${CMAKE_SOURCE_DIR}/par_v7_adapted_for_picador/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v7_adapted_for_picador/Vectors.h
    ${CMAKE_SOURCE_DIR}/par_v7_adapted_for_picador/Particle.h
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    )
target_compile_definitions(QED_benchmark_v7 PUBLIC __VER_QED_BENCHMARK__="v7" PUBLIC __CHUNK_SIZE__=128 PUBLIC __ITER_NUM__=${ITER} PUBLIC __ITER_DUMP__=${ITER_DUMP})
target_include_directories(QED_benchmark_v7 PUBLIC ${CMAKE_SOURCE_DIR}/par_v7_adapted_for_picador/)

add_executable(QED_benchmark_v8
    ${CMAKE_SOURCE_DIR}/par_v8_adapted_for_picador_simpler/macros.h
    ${CMAKE_SOURCE_DIR}/par_v8_adapted_for_picador_simpler/breit_wheeler.h
    ${CMAKE_SOURCE_DIR}/par_v8_adapted_for_picador_simpler/compton.h
    ${CMAKE_SOURCE_DIR}/par_v8_adapted_for_picador_simpler/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v8_adapted_for_picador_simpler/Vectors.h
    ${CMAKE_SOURCE_DIR}/par_v8_adapted_for_picador_simpler/Particle.h
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    )
target_compile_definitions(QED_benchmark_v8 PUBLIC __VER_QED_BENCHMARK__="v8" PUBLIC __CHUNK_SIZE__=128 PUBLIC __ITER_NUM__=${ITER} PUBLIC __ITER_DUMP__=${ITER_DUMP})
target_include_directories(QED_benchmark_v8 PUBLIC ${CMAKE_SOURCE_DIR}/par_v8_adapted_for_picador_simpler/)

add_executable(QED_benchmark_v9
    ${CMAKE_SOURCE_DIR}/par_v9_adapted_for_picador_delete_died_photons/macros.h
    ${CMAKE_SOURCE_DIR}/par_v9_adapted_for_picador_delete_died_photons/breit_wheeler.h
    ${CMAKE_SOURCE_DIR}/par_v9_adapted_for_picador_delete_died_photons/compton.h
    ${CMAKE_SOURCE_DIR}/par_v9_adapted_for_picador_delete_died_photons/Fast_QED.h
    ${CMAKE_SOURCE_DIR}/par_v9_adapted_for_picador_delete_died_photons/Vectors.h
    ${CMAKE_SOURCE_DIR}/par_v9_adapted_for_picador_delete_died_photons/Particle.h
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    )
target_compile_definitions(QED_benchmark_v9 PUBLIC __VER_QED_BENCHMARK__="v9" PUBLIC __CHUNK_SIZE__=128 PUBLIC __ITER_NUM__=${ITER} PUBLIC __ITER_DUMP__=${ITER_DUMP})
if (RAND_SEED)
    target_compile_definitions(QED_benchmark_v9 PUBLIC __RAND_SEED__=1)
endif()
target_include_directories(QED_benchmark_v9 PUBLIC ${CMAKE_SOURCE_DIR}/par_v9_adapted_for_picador_delete_died_photons/)
