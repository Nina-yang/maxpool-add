# python generate_test_case.py -s small -o ./test_case_small
# python generate_test_case.py -s mid -o ./test_case_mid
# python generate_test_case.py -s large -o ./test_case_large



# cd build
# cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_OMP=ON -DUSE_AVX=ON
# make 

# cd build
# cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_OMP=OFF -DUSE_AVX=ON
# make 

# cd build
# cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_OMP=ON -DUSE_AVX=OFF
# make 

# cd build
# cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_OMP=OFF -DUSE_AVX=OFF
# make 

# cd ..
epoch=10
####################################################
# omp off, avx off
####################################################
# for ((i=1; i<=$epoch; i++))
# do
# ./build/mpa ./test_case_small >> report/mpa_small.txt
# done

# for ((i=1; i<=$epoch; i++))
# do
# ./build/mpa ./test_case_mid >> report/mpa_mid.txt
# done

# for ((i=1; i<=$epoch; i++))
# do
# ./build/mpa ./test_case_large >> report/mpa_large.txt
# done


####################################################
# omp on, avx off
####################################################
# for ((i=1; i<=$epoch; i++))
# do
# ./build/mpa_omp ./test_case_small >> report/mpa_omp_small.txt
# done

# for ((i=1; i<=$epoch; i++))
# do
# ./build/mpa_omp ./test_case_mid >> report/mpa_omp_mid.txt
# done

# for ((i=1; i<=$epoch; i++))
# do
# ./build/mpa_omp ./test_case_large >> report/mpa_omp_large.txt
# done


####################################################
# omp off, avx on
####################################################
# for ((i=1; i<=$epoch; i++))
# do
# ./build/mpa_avx ./test_case_small >> report/mpa_avx_small.txt
# done

# for ((i=1; i<=$epoch; i++))
# do
# ./build/mpa_avx ./test_case_mid >> report/mpa_avx_mid.txt
# done

# for ((i=1; i<=$epoch; i++))
# do
# ./build/mpa_avx ./test_case_large >> report/mpa_avx_large.txt
# done


####################################################
# omp on, avx on
####################################################
# for ((i=1; i<=$epoch; i++))
# do
# ./build/mpa_omp_avx ./test_case_small >> report/mpa_omp_avx_small.txt
# done

# for ((i=1; i<=$epoch; i++))
# do
# ./build/mpa_omp_avx ./test_case_mid >> report/mpa_omp_avx_mid.txt
# done

# for ((i=1; i<=$epoch; i++))
# do
# ./build/mpa_omp_avx ./test_case_large >> report/mpa_omp_avx_large.txt
# done


########################
# generate report
########################
python generate_report.py