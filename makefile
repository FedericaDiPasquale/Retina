###############################################################################
#                                                                             #
#                                 makefile                                    #
#                                                                             #
#                        Author: Federica Di Pasquale                         #
#                                                                             #
###############################################################################

# output filename
NAME = monk

# basic directory
DIR = ./

# Eigen include directory
libEigenINC = -I/usr/local/Cellar/eigen/3.3.9/include/eigen3/

# Gnuplot include directory
gnuINC = -I/Users/federicadipasquale/gnuplot-iostream
gnuLIB = -lboost_iostreams -lboost_system -lboost_filesystem

# compiler
CC = clang++

# compiler flags
CFLAGS = -O3 -std=c++17 

# -----------------------------------------------------------------------------

$(NAME): $(DIR)$(NAME).cpp $(DIR)NeuralNetwork.cpp $(DIR)NeuralNetwork.hpp
	$(CC) -o $(NAME) $(DIR)$(NAME).cpp $(DIR)NeuralNetwork.cpp \
        $(libEigenINC) $(gnuINC) $(gnuLIB) $(CFLAGS)

# clean -----------------------------------------------------------------------

.PHONY: clean
clean:
	rm -f $(DIR)*.o $(DIR)*~ $(NAME)

###############################################################################
