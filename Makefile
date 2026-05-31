COMPILER = nvcc
EXES = prime2

all: $(EXES)

prime2: prime2.cu
	$(COMPILER) -O3 -std=c++17 -arch=sm_80 prime2.cu -o prime2

run: $(EXES)
	./$(EXES) 1000000 20000000 3 512 512

clean:
	rm -f $(EXES)
	rm -f iter_*.bmp iter_*.csv *.bmp *.csv
	rm -f all_iterations_summary.csv
	rm -f run_stdout.txt run_time.txt prime_report_output.txt deep_launcher.log
	rm -rf deep_quad_results_* auto_explore_* zoom_* zoom_quad_* test_chunk chunk_*