TOPDIR=..
include $(TOPDIR)/Makefile.include

MAKEDEPEND= $(CC) -M $(CXXFLAGS) $(CPPFLAGS) $(BSMINCL) $(GBLASINCL) $(CUDAUTILSINCL) -o $*.d $<
#SRCS = test_matrix_creation.cc test_matrix_operations.cc
SRCS = test_matrix_creation.cc

#all: test_matrix_creation test_matrix_operations test_matrix_alloc_problem
all: test_matrix_creation

# test_matrix_creation
test_matrix_creation: test_matrix_creation.o test_utils.o
	$(CC) $(CXXFLAGS) $(CPPFLAGS) $(BSMINCL) $(GBLASINCL) $(CUDAUTILSINCL) -o $@ test_matrix_creation.o test_utils.o $(CUDAUTILSDIR)/libcudautils.a $(BLASFLAGS)

# test_matrix_operations
#test_matrix_operations: test_matrix_operations.o test_utils.o
#	$(CC) $(CXXFLAGS) $(CPPFLAGS) $(BSMINCL) $(GBLASINCL) $(CUDAUTILSINCL) -o $@ test_matrix_operations.o test_utils.o $(CUDAUTILSDIR)/libcudautils.a $(BLASFLAGS) $(LDFLAGS)

# test_matrix_alloc_problem
#test_matrix_alloc_problem: test_matrix_alloc_problem.o test_utils.o
#	$(CC) $(CXXFLAGS) $(CPPFLAGS) $(BSMINCL) $(GBLASINCL) $(CUDAUTILSINCL) -o $@ test_matrix_alloc_problem.o test_utils.o $(CUDAUTILSDIR)/libcudautils.a $(BLASFLAGS) #$(LDFLAGS)

%.o: %.cc
	@$(MAKEDEPEND)
	$(CC) $(CXXFLAGS) $(CPPFLAGS) $(BSMINCL) $(GBLASINCL) $(CUDAUTILSINCL) -c $< -o $@

-include $(SRCS:.cc=.d)

check: all
	./test_matrix_creation
#	./test_matrix_operations
#	./test_matrix_alloc_problem 7 1 1 1 1

clean:
	rm -f test_matrix_creation #test_matrix_operations test_matrix_alloc_problem *.o *.d
