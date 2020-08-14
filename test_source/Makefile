TOPDIR=..
include $(TOPDIR)/Makefile.include

MAKEDEPEND= $(CC) -M $(CXXFLAGS) $(CPPFLAGS) $(HBSMINCL) -o $*.d $<
SRCS = test_matrix_creation.cc test_matrix_operations.cc

all: test_matrix_creation test_matrix_operations test_matrix_alloc_problem

# test_matrix_creation
test_matrix_creation: test_matrix_creation.o test_utils.o
	$(CC) $(CXXFLAGS) $(CPPFLAGS) $(HBSMINCL) -o $@ test_matrix_creation.o test_utils.o  $(BLASFLAGS)

# test_matrix_operations
test_matrix_operations: test_matrix_operations.o test_utils.o
	$(CC) $(CXXFLAGS) $(CPPFLAGS) $(HBSMINCL) -o $@ test_matrix_operations.o test_utils.o  $(BLASFLAGS) $(LDFLAGS)

# test_matrix_alloc_problem
test_matrix_alloc_problem: test_matrix_alloc_problem.o test_utils.o
	$(CC) $(CXXFLAGS) $(CPPFLAGS) $(HBSMINCL) -o $@ test_matrix_alloc_problem.o test_utils.o  $(BLASFLAGS) $(LDFLAGS)

%.o: %.cc
	@$(MAKEDEPEND)
	$(CC) $(CXXFLAGS) $(CPPFLAGS) $(HBSMINCL) -c $< -o $@

-include $(SRCS:.cc=.d)

check: all
	./test_matrix_creation
	./test_matrix_operations
	./test_matrix_alloc_problem 32 4 20 20 1

clean:
	rm -f test_matrix_creation *.o *.d test_matrix_operations test_matrix_alloc_problem 
