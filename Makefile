# Adaptive Fixed-Point kNN Engine - build file
CC      = gcc
CFLAGS  = -std=c99 -Wall -Wextra -Wpedantic -O2
SRC     = src/dataset.c src/normalize.c src/knn.c src/metrics.c src/aqi.c

all: knn_engine

knn_engine: $(SRC) src/main.c src/*.h
	$(CC) $(CFLAGS) -o $@ $(SRC) src/main.c -lm


test: tests/test_knn.c $(SRC) src/*.h
	$(CC) $(CFLAGS) -Isrc -o test_knn tests/test_knn.c $(SRC) -lm
	./test_knn

# debug build with the compiler's memory / undefined-behaviour checkers
debug: $(SRC) src/main.c src/*.h
	$(CC) -std=c99 -Wall -Wextra -Wpedantic -g -fsanitize=address,undefined -o knn_engine_debug $(SRC) src/main.c -lm

demo: knn_engine
	./knn_engine data/real_training.csv --demo

clean:
	rm -f knn_engine knn_engine_debug test_knn

.PHONY: all test debug demo clean
