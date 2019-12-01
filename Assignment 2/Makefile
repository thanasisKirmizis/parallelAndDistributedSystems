CC = gcc-7
MPICC = mpicc
AR = ar
CFLAGS  = -g -c
RM = rm -f
LIBFLAGS = -lm -lopenblas -lpthread
TARGET1 = knnring_sequential
TARGET2 = knnring_mpi

#Here write your path to the include directories of OpenBLAS
INCLUDES= -I /opt/OpenBLAS/include -L/opt/OpenBLAS/lib

OUT_DIR=./lib
SRC_DIR=./src
	
lib: $(TARGET1) $(TARGET2)

#Make library for Sequential
$(TARGET1): $(SRC_DIR)/$(TARGET1).c
	$(CC) $(CFLAGS) $(LIBFLAGS) $(INCLUDES) $(SRC_DIR)/$(TARGET1).c -o $(OUT_DIR)/$(TARGET1).o 
	$(AR) rcs $(OUT_DIR)/$(TARGET1).a $(OUT_DIR)/$(TARGET1).o
	$(RM) $(OUT_DIR)/$(TARGET1).o

#Make library for MPI
$(TARGET2): $(SRC_DIR)/$(TARGET2).c
	$(MPICC) $(CFLAGS) $(LIBFLAGS) $(INCLUDES) $(SRC_DIR)/$(TARGET2).c -o $(OUT_DIR)/$(TARGET2).o 
	$(AR) rcs $(OUT_DIR)/$(TARGET2).a $(OUT_DIR)/$(TARGET2).o
	$(RM) $(OUT_DIR)/$(TARGET2).o

# clean temporary files
clean:
	$(RM) *.o *~
