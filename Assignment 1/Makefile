CC = gcc-7
AR = ar
CFLAGS  = -g -c
RM = rm -f
LIBFLAGS = -lm -lpthread -fopenmp -fcilkplus
TARGET1 = vptree_sequential
TARGET2 = vptree_pthreads
TARGET3 = vptree_cilk
TARGET4 = vptree_openmp

INCLUDES= -I ./inc

OUT_DIR=./lib
SRC_DIR=./src
	
lib: $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4)

#Make library for Sequential
$(TARGET1): $(SRC_DIR)/$(TARGET1).c
	$(CC) $(CFLAGS) $(LIBFLAGS) $(INCLUDES) $(SRC_DIR)/$(TARGET1).c -o $(OUT_DIR)/$(TARGET1).o 
	$(AR) rcs $(OUT_DIR)/$(TARGET1).a $(OUT_DIR)/$(TARGET1).o
	$(RM) $(OUT_DIR)/$(TARGET1).o

#Make library for Pthreads
$(TARGET2): $(SRC_DIR)/$(TARGET2).c
	$(CC) $(CFLAGS) $(LIBFLAGS) $(INCLUDES) $(SRC_DIR)/$(TARGET2).c -o $(OUT_DIR)/$(TARGET2).o
	$(AR) rcs $(OUT_DIR)/$(TARGET2).a $(OUT_DIR)/$(TARGET2).o
	$(RM) $(OUT_DIR)/$(TARGET2).o

#Make library for Cilk
$(TARGET3): $(SRC_DIR)/$(TARGET3).c
	$(CC) $(CFLAGS) $(LIBFLAGS) $(INCLUDES) $(SRC_DIR)/$(TARGET3).c -o $(OUT_DIR)/$(TARGET3).o 
	$(AR) rcs $(OUT_DIR)/$(TARGET3).a $(OUT_DIR)/$(TARGET3).o
	$(RM) $(OUT_DIR)/$(TARGET3).o

#Make library for OpenMP
$(TARGET4): $(SRC_DIR)/$(TARGET4).c
	$(CC) $(CFLAGS) $(LIBFLAGS) $(INCLUDES) $(SRC_DIR)/$(TARGET4).c -o $(OUT_DIR)/$(TARGET4).o 
	$(AR) rcs $(OUT_DIR)/$(TARGET4).a $(OUT_DIR)/$(TARGET4).o
	$(RM) $(OUT_DIR)/$(TARGET4).o

# clean temporary files
clean:
	$(RM) *.o *~
