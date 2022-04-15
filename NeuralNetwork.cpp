/****************************************************************************/
/*                            NeuralNetwork.cpp                             */
/****************************************************************************/

#include "NeuralNetwork.hpp"

/****************************************************************************/
// Given data X and corresponding target(s) y, train the model

void MultiLayerPerceptron::train(Matrix X, Matrix y) {

    // Check dimensions
    if ( X.rows() != y.rows() )
        throw(std::invalid_argument("X.rows() must be = y.rows()"));

    /*************************** Initializations ****************************/

    topology.front() = X.cols();  // number of attributes        
    topology.back() = y.cols();   // number of outputs         
    
    vec_Matrix dW;                // matrices of delta weights (= -gradient)
    vec_Matrix Xb, yb;            // vectors of blocks of data for mini-batch

    // Initiliaze W with small random values and dW with all 0's
    initialize_weights(dW);                        
  
    learning_curve.clear();

    int n_iter = 0;                 

    /****************************** MAIN LOOP *******************************/

    while ( true ) {

        shuffle(X, y, rg);          // shuffle data at each iteration           

        double error = 0;           // initialize error
        
        // Split X and y in blocks of size = batch_size     
        split(X, y, Xb, yb, batch_size);      

        // for each block of data
        for ( int i = 0; i < Xb.size(); ++i ) {

            // compute output for current data Xb[i]
            Matrix y = forward_pass(Xb[i]);      

            // update error w.r.t. to current targets yb[i]
            error += MSE(yb[i], y);                 

            // compute delta = (target - output) * derivative(output)
            Matrix delta = (yb[i] - y).cwiseProduct(f(get_n_layers()-1,1));
            
            // backpropagate delta to compute dW
            backward_pass(delta, dW);            

            // update weights adding dW and regularization (if any)
            update_weights(dW);

        }
        
        learning_curve.push_back(error);           

        /*********************** Stopping Conditions ************************/

        if ( error < tol )                  // Tolerance condition
            break;

        if ( ++n_iter == max_iter )         // Iteration limit
            break;
    
    } // end main loop

} // end MultiLayerPerceptron::train()

/****************************************************************************/
// Initialize matrices of weights with small random values

void MultiLayerPerceptron::initialize_weights(vec_Matrix &dW) {

    // A matrix of weights between each pair of adjacent layers
    W.resize(get_n_layers() - 1);
    dW.resize(get_n_layers() - 1);
    
    // Generate matrices with small random values in (-0.7,0.7)
    for ( int i = 0; i < W.size(); ++i ) {
        int m = topology[i+1];             // number of rows              
        int n = topology[i] + 1;           // number of columns (+1 for bias)           
        W[i] = random_matrix(m, n, -0.7, 0.7, rg);
        dW[i] = Matrix::Zero(m, n);                 
    }
} 

/****************************************************************************/
// Compute outputs for input data X

Matrix MultiLayerPerceptron::forward_pass(const Matrix &X) {
    
    nets[0] = X;        
    for ( int i = 1; i < get_n_layers(); ++i ) 
        nets[i] = (f(i-1) + bias) * W[i-1].transpose();

    return f(get_n_layers()-1);
}

/****************************************************************************/
// Backpropagate delta and compute delta weights

void MultiLayerPerceptron::backward_pass(Matrix &delta, vec_Matrix &dW) {

    for ( int i = get_n_layers() - 2; i >= 0; --i ) {
        dW[i] = delta.transpose() * (f(i) + bias);
        delta = (delta * (W[i] - bias)).cwiseProduct(f(i,1));
    }
}

/****************************************************************************/
// Update weights also applying regularization (if any)

void MultiLayerPerceptron::update_weights(vec_Matrix &dW) {

    for ( int i = 0; i < W.size(); ++i ) {

        W[i] += eta * dW[i]             // Learning Rate
              - lambda * W[i];          // Regularization

        // Re-initialize dW
        dW[i] = Matrix::Zero(dW[i].rows(), dW[i].cols());
    }
}

/*************************** Activation functions ***************************/
// Apply activation function sf to x. df = 1 for first derivative

double MultiLayerPerceptron::f(double x, std::string sf, int df) {

    /******************************* sigmoid ********************************/
    if ( sf == "sigmoid" ) {
        switch ( df ) {
            case 0: return 1.0 / (1 + exp(-x));
            case 1: return  f(x,sf) * (1 - f(x,sf));
            default: throw(std::invalid_argument("df >= 2 not supported"));
        }
    }
    /******************************* identity *******************************/    
    else if ( sf == "identity" ) {
        switch ( df ) {
            case 0: return x;
            case 1: return 1;
            default: throw(std::invalid_argument("df >= 2 not supported"));
        }
    } 
    /********************************* tanh *********************************/
    else if ( sf == "tanh" ){
        switch ( df ) {
            case 0: return tanh(x);
            case 1: return 1 - tanh(x) * tanh(x);
            default: throw(std::invalid_argument("df >= 2 not supported"));
        }
    }
    else
        throw(std::invalid_argument("Wrong activation function"));
}

/****************************************************************************/
// Return f(net[layer]). df = 1 for derivative

Matrix MultiLayerPerceptron::f(int layer, int df) {

    std::string sf;         // string to specify the activation function

    if ( layer == 0 )
        sf = "identity";
    else if ( layer == get_n_layers() - 1 )
        sf = f_out;
    else
        sf = f_hid;

    // Initialize result
    Matrix fnet(nets[layer].rows(), nets[layer].cols());      

    for ( int i = 0; i < fnet.rows(); ++i )
        for( int j = 0; j < fnet.cols(); ++j )
            fnet(i,j) = f(nets[layer](i,j), sf, df);

    return fnet;
}

/***************************** Private Methods ******************************/
// Shuffle data X and terget(s) y 

void MultiLayerPerceptron::shuffle(Matrix &X, Matrix &y, rand_gen rg) {

    Eigen::VectorXi idx = Eigen::VectorXi::LinSpaced(X.rows(), 0, X.rows());

    std::shuffle(idx.data(), idx.data() + X.rows(), rg);

    X = idx.asPermutation() * X;
    y = idx.asPermutation() * y;

}

/****************************************************************************/
// Split data in blocks of size = batch_size and store them in Xb and yb

void MultiLayerPerceptron::split(Matrix &X, Matrix &y, 
                                 vec_Matrix &Xb, vec_Matrix &yb, 
                                 int batch_size) { 

    int n = X.rows() / batch_size;
    int mod = X.rows() % batch_size;

    // Initialize result
    Xb.resize(n + 1);
    yb.resize(n + 1);

    int idx = 0;
    for ( int i = 0; i < n; ++i ) {
        Xb[i] = X.block(idx, 0, batch_size, X.cols());
        yb[i] = y.block(idx, 0, batch_size, y.cols());
        idx += batch_size;
    }

    if ( mod == 0 ) {
        Xb.resize(n);
        yb.resize(n);
    }
    else {
        Xb.back() = X.block(idx, 0, mod, X.cols());
        yb.back() = y.block(idx, 0, mod, y.cols());
    }
}

/****************************************************************************/
// return a random matrix m x n with values v in l <= v <= u

Matrix MultiLayerPerceptron::random_matrix(int m, int n,  
                                           double l, double u, rand_gen rg) {
    std::uniform_real_distribution<double> dis(l, u);
    return Matrix::Zero(m,n).unaryExpr([&](double dummy){return dis(rg);});
}

/****************************************************************************/
// return mean square error

double MultiLayerPerceptron::MSE(const Matrix &target, const Matrix &output) 
{    
    Matrix delta = target - output;

    double mse = 0;     // Initialize result

    // for each pattern
    for ( int p = 0; p < delta.rows(); ++p )
        mse += (delta.row(p).array().square().sum()) / 2;

    return mse;
}

/****************************************************************************/
// Given a Matrix X, add a column of 1's for the bias

Matrix operator + (Matrix X, Bias b) {
    X.conservativeResize(X.rows(), X.cols() + 1);
    b.resize(X.rows());
    X.rightCols(1) = b.col1;
    return X;
}

/****************************************************************************/
// Given a Matrix X, remove last column to remove the bias

Matrix operator - (Matrix X, Bias b) {
    X.conservativeResize(X.rows(), X.cols() - 1);
    return X;
}

/****************************************************************************/