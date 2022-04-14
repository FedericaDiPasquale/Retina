/****************************************************************************/
/*                            NeuralNetwork.hpp                             */
/****************************************************************************/
/* Implementation of a feedforward fully-connected Neural Network.
* 
* Two main classes are defined: MultiLayerPerceptron Class which can be used
* to perform a Regression task; BinaryClassifier Class (derived from
* MultiLayerPerceptron) to perform a Binary Classification task. 
*/

#ifndef NEURAL_NETWORK_HPP_         
#define NEURAL_NETWORK_HPP_ 

/****************************************************************************/

#include <random>
#include <string>
#include <vector>
#include <Eigen/Dense>

typedef Eigen::MatrixXd Matrix;
typedef std::vector<Matrix> vec_Matrix;
typedef std::default_random_engine rand_gen;

/*********************************** Bias ***********************************/

class Bias {
public:
    void resize(int dim) { col1 = Matrix::Constant(dim, 1, 1); }
    Matrix col1;
};

// Given a Matrix X, add a column of 1's for the bias
Matrix operator + (Matrix X, Bias b);

// Given a Matrix X, remove last column to remove the bias
Matrix operator - (Matrix X, Bias b);

/************************* MLPConfiguration struct **************************/

typedef struct {
    int max_iter;                          // max number of iteration
    int seed;                              // seed for the random generator
    int batch_size;                        // batch size
    double alpha;                          // momentum
    double eta;                            // learning rate
    double lambda;                         // regularization
    double tol;                            // tolerance for the optimization
    std::string f_hid;                     // act_function hidden layers
    std::string f_out;                     // act_function output layer
    std::vector<int> hidden_layer_sizes;   // topology hidden layers
} MLPConfiguration;

/*********************** MultiLayerPerceptron CLASS *************************/

class MultiLayerPerceptron {

public:

    MultiLayerPerceptron() : max_iter(3000), seed(999), batch_size(1), 
                             alpha(0.0), eta(0.01), lambda(0.0), 
                             f_hid("sigmoid"), f_out("identity"), tol(1e-4), 
                             topology({0, 5, 0})
                             { rg.seed(seed); nets.resize(get_n_layers()); }

    ~MultiLayerPerceptron() {};

    /**************************** Public Methods ****************************/
    // Given data X and corresponding target(s) y, train the model

    void train(Matrix X, Matrix y);

    /************************************************************************/
    // Given a matrix of data X, compute and return the predicted outputs

    virtual Matrix predict(const Matrix &X) { return forward_pass(X); }

    /*************************** Getters and Setters ************************/
        
    int get_n_layers() { return topology.size(); }
    
    int get_seed() { return seed; }
    
    int get_max_iter() { return max_iter; }
    
    int get_batch_size() { return batch_size; }
    
    double get_eta() { return eta; }
    
    double get_alpha() { return alpha; }
    
    double get_lambda() { return lambda; }
    
    double get_tolerance() { return tol; }

    std::string get_f_hid() { return f_hid; }

    std::string get_f_out() { return f_out; }

    std::vector<double> & get_learning_curve() { return learning_curve; }

    /************************************************************************/
        
    void set_seed(int seed) { this->seed = seed; rg.seed(seed); }
    
    void set_max_iter(int max_iter) { this->max_iter = max_iter; }
    
    void set_batch_size(int batch_size) { this->batch_size = batch_size; }
    
    void set_eta(double eta) { this->eta = eta; }
    
    void set_alpha(double alpha) { this->alpha = alpha; }
    
    void set_lambda(double lambda) { this->lambda = lambda; }
    
    void set_tolerance(double tol) { this->tol = tol; }

    void set_f_hid(std::string f_hid) { this->f_hid = f_hid; }

    void set_f_out(std::string f_out) { this->f_out = f_out; }

    void set_hidden_layer_sizes(std::vector<int> hls) { 
        
        topology.resize(hls.size() + 2);
        for ( int i = 0; i < hls.size(); ++i )
            topology[i+1] = hls[i];
        
        topology.front() = 0;
        topology.back() = 0;
        
        nets.resize(get_n_layers());
    }

    /************************************************************************/
    // Configure parameters

    void configure(MLPConfiguration &params) {
        set_max_iter(params.max_iter);
        set_seed(params.seed);
        set_batch_size(params.batch_size);
        set_alpha(params.alpha);
        set_eta(params.eta);
        set_lambda(params.lambda);
        set_tolerance(params.tol);
        set_f_hid(params.f_hid);
        set_f_out(params.f_out);
        set_hidden_layer_sizes(params.hidden_layer_sizes);
    } 

protected:

    /************************** Protected Methods ***************************/

    void initialize_weights(vec_Matrix &dW);

    Matrix forward_pass(const Matrix &X);

    void backward_pass(Matrix &delta, vec_Matrix &dW);

    void update_weights(vec_Matrix &dW);

    /************************* Activation functions *************************/

    double f(double x, std::string sf = "", int df = 0);

    Matrix f(const Matrix &X, std::string sf = "", int df = 0);

    /*************************** Protected Fields ***************************/

    vec_Matrix nets;                            // Net of each unit
    vec_Matrix W;                               // Weights
    std::vector<int> topology;                  // Topology
    std::vector<double> learning_curve;         // Errors at each iteration
    rand_gen rg;                                // Random Generator
    Bias bias;                                  // Column of 1's

    /****************************** PARAMETERS ******************************/
    
    int seed;                       // seed for random genetaror
    int max_iter;                   // max number of iteration
    int batch_size;                 // batch size

    double eta;                     // learning rate
    double tol;                     // tolerance for the optimization
    double alpha;                   // momentum
    double lambda;                  // regularization parameter

    std::string f_hid;              // activation function hidden layers
    std::string f_out;              // activation function output layer

private:

    /*************************** Private Methods ****************************/

    // Shuffle data X and terget(s) y 
    void shuffle(Matrix &X, Matrix &y, rand_gen rg);

    // split data for mini-batch
    void split(Matrix &X, Matrix &y, vec_Matrix &Xb, vec_Matrix &yb, 
               int batch_size);

    // return a random matrix m x n with values v in l <= v <= u
    Matrix random_matrix(int m, int n,  double l, double u, rand_gen rg);

    // return mean square error
    double MSE(const Matrix &target, const Matrix &output);

}; // end MultiLayerPerceptron class

/************************** BinaryClassifier CLASS **************************/

class BinaryClassifier : public MultiLayerPerceptron {

public:

    BinaryClassifier() : MultiLayerPerceptron() {}

    ~BinaryClassifier() {}

    /**************************** Public Methods ****************************/
    // Given a matrix of data X, compute and return the predicted output(s)

    Matrix predict(const Matrix &X) override {

        Matrix y = MultiLayerPerceptron::predict(X);

        // Apply treshold
        for ( int i = 0; i < y.rows(); ++i )
            for ( int j = 0; j < y.cols(); ++j )
                y(i,j) = y(i,j) > 0.5 ? 1 : 0;
        
        return y;
    }

    /************************************************************************/
    // Given X data and y target(s) return Accuracy

    double accuracy(const Matrix &X, const Matrix &y) {

        Matrix y_predicted = predict(X);

        // Count miss_classified patterns
        int miss_classified = 0;
        for ( int p = 0; p < X.rows(); ++p ) {
            if ( y_predicted(p) != y(p) )
                miss_classified++;
        }

        return double(X.rows() - miss_classified) / double(X.rows());
    }

protected:

    /*************************** Protected Fields ***************************/

    std::vector<double> accuracy_curve;         // Accuracy at each iteration

}; // end BinaryClassifier class

#endif // NEURAL_NETWORK_HPP_ 

/****************************************************************************/