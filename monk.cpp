/****************************************************************************/
/*                                 Monk.cpp                                 */
/****************************************************************************/

#include <iostream>
#include <fstream>
#include "gnuplot-iostream.h"
#include "NeuralNetwork.hpp"

/*************************** Auxiliary functions ****************************/
// Implementations at the end of the file

void readfile(Matrix &X, Matrix &y, std::string file_path);

void plot(const std::vector<double> v);

/*********************************** MAIN ***********************************/

int main(int argc, char **argv) {

    // Command line parameter

    if ( argc < 2 ) {
        std::cerr << "Usage " << argv[0] << " [1|2|3]" << std::endl;
        return 1;
    }

    std::string monk = argv[1];

    /************************* Store Data from file *************************/

    Matrix X_train, y_train, X_test, y_test;

    readfile(X_train, y_train, "./data/monks-" + monk + ".train");

    readfile(X_test, y_test, "./data/monks-" + monk + ".test");

    /************************************************************************/
    
    BinaryClassifier clf;

    /****************************** Parameters ******************************/

    MLPConfiguration params;

    params.max_iter = 2000;                 // max number of iteration
    params.seed = 999;                      // seed for random generator
    params.batch_size = X_train.rows();     // batch size
    params.alpha = 0.0;                     // momentum
    params.eta = 0.05;                      // learning rate
    params.lambda = 0.0;                    // regularization
    params.tol = 1e-4;                      // tolerance
    params.f_hid = "sigmoid";               // act_function hidden layers
    params.f_out = "sigmoid";               // act_function output layer
    params.hidden_layer_sizes = {10};       // 10 units in 1 hidden layer

    clf.configure(params);                  // Configure classifier

    /**************************** Train and Test ****************************/

    clf.train(X_train, y_train);

    std::cout << "Accuracy on training set: " 
              << clf.accuracy(X_train, y_train) * 100 << "%"
              << std::endl;

    std::cout << "Accuracy on test set: " 
              << clf.accuracy(X_test, y_test) * 100 << "%"
              << std::endl;

    plot(clf.get_learning_curve());

    return 0;
}

/*************************** Auxiliary functions ****************************/

void readfile(Matrix &X, Matrix &y, std::string file_path) {
    
    /******************** Format of Monks' datasets: ************************/
    /*  | target | attr1 | attr2 | attr3 | attr4 | attr5 | attr 6 | skip |  */
    /************************************************************************/

    // Open the file
    std::ifstream file(file_path);

    // Count number of lines (patterns)
    int n_patterns = 0;
    std::string skip;
    while ( getline(file, skip) )
        n_patterns++;
    
    file.clear();
    file.seekg(0);

    X.resize(n_patterns, 6);
    y.resize(n_patterns, 1);

    /*********************** Read content of the file ***********************/

    for ( int i = 0; i < n_patterns; ++i ) {
        
        file >> y(i);                   // target 
        
        for ( int j = 0; j < 6; ++j )
            file >> X(i,j);             // attributes

        file >> skip;                   // skip last string
    }

    file.close();

}

/****************************************************************************/

void plot(const std::vector<double> v) {

    Gnuplot gp;

    // points to plot
    std::vector<std::pair<int, double>> pts;

    for ( int i = 0; i < v.size(); ++i )
        pts.push_back(std::make_pair(i, v[i]));
    
    gp << "plot" << gp.file1d(pts) << "with lines title 'Learning Curve'" 
       << std::endl;

} 

/****************************************************************************/