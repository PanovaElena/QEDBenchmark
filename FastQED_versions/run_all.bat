@echo off

set target_list=QED_benchmark_v0 QED_benchmark_v1 QED_benchmark_v2_1 QED_benchmark_v2_2 QED_benchmark_v2_3 QED_benchmark_v2_4 QED_benchmark_v3_1  QED_benchmark_v3_2 QED_benchmark_v3_3 QED_benchmark_v3_4 QED_benchmark_v3_5

for %%t in (%target_list%) do (
    for /l %%i in (1, 1, 5) do (
        echo %%t
        .\%%t.exe
    )
)
