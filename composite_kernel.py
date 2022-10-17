from sklearn.gaussian_process.kernels import *
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score, learning_curve
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real
from sklearn.svm import SVC
from data_extraction import get_X_y, get_subset_n, get_training_test_data
data_dir = "datasets/"

CONST = ConstantKernel(constant_value=1,constant_value_bounds=(0.1,100))
DEFAULT_BASIS = [RBF(length_scale=1,length_scale_bounds=(0.01,100)),
                 Matern(length_scale=1, length_scale_bounds=(0.01,100),nu=0.5),
                 Matern(length_scale=1, length_scale_bounds=(0.01,100),nu=1.5),
                 Matern(length_scale=1, length_scale_bounds=(0.01,100),nu=2.5),
                 RationalQuadratic(length_scale=1,alpha=1,length_scale_bounds=(0.01,100),alpha_bounds=(0.01,100))
                 ]
                #  DotProduct(sigma_0=2e-7,sigma_0_bounds="fixed")
#TODO: include polynomial kernel

def build_composite_kernel(X, y, n_layers, kernel_basis=None):
    """
    Build composite kernel for regression problem with training data X and y
    Args:
        X: np.array, design matrix of size (n_training_points x dim)
        y: np.array, target training vector of size (n_training_points,)
        n_layer: int, number of layers for the composite kernel
        kernel_basis: list[Kernel], list containing basis kernels

    Returns:
        list[Kernel], best kernels at every layer
    """
    if kernel_basis is None:
        kernel_basis = DEFAULT_BASIS
    if n_layers == 1:
        best_kernels = [CONST*choose_best_kernel(kernel_basis,X,y)[0]]
    else:
        best_kernels_prev = build_composite_kernel(X,y,n_layers-1,kernel_basis)
        kernel_combinations = generate_possible_kernels(best_kernels_prev[-1],kernel_basis)
        best_kernels_prev.append(choose_best_kernel(kernel_combinations,X,y)[0])
        best_kernels = best_kernels_prev
    print(f'Done layer {n_layers}, best kernel so far: {best_kernels[-1]}')
    return best_kernels

def choose_best_kernel(kernel_list,X,y):
    """
    Take a list of kernels and return the best kernel base on MLH and BIC on a training set
    Args:
        kernel_list: list[Kernel], list of kernels for consideration
        X: np.array, design matrix of size (dim x n_training_points)
        y: np.array, target training vector of size (n_training_points,)

    Returns:
        best_kernel: Kernel, best kernel overall in the list
        BIC: float, Bayesian Information Criterion of the best kernel
        Log_loss: float, log loss of the best kernel
    """
    #TODO: Parallelize this operation
    result_list = [None,]*len(kernel_list)
    BIC_list = [0,]*len(kernel_list)

    for i in range(len(kernel_list)):
        kernel = kernel_list[i]
        results = optimize_kernel(kernel,X,y)
        result_list[i] = results
        BIC_list[i] = results[1]

    best_kernel_index = BIC_list.index(min(BIC_list))
    return result_list[best_kernel_index]

def get_learning_curves_in_k(kernel, n, k_range, n_folds, training_sizes, kernel_ID, optimize=False,
                             composite=False, kernel_basis=None, n_layer=4, optimize_training_size=5000):
    """
    Get learning curves of SVM with given kernel with increasing k in the k-forrelation set
    Args:
        kernel: Kernel, kernel template for valuation (might be optimized)
        n: int, length of input bit string of k-forrelation
        k_range: list[int] or int, [min_k, max_k] with min_k and max_k odd and step = 2, or just one k value
        n_folds: int, number of fold for cross-validation
        training_sizes: list[int], list of step training sizes for the learning curve
        optimize: boolean, if True, first optimize kernel with BO using optimize_training_size points
        composite: boolean, if True, build composite kernel based on kerel_basis and disregard given kernel
        kernel_basis: list[Kernel], list containing basis kernels
        n_layer: int, number of layer for composite kernel
        optimize_training_size: int, number of training points use for BO optimization
                                    neglected if optimize = False
        kernel_ID: str, int, ID for the kernel for writing to file

    Returns:
        test_scores_list: np.array, of size (number of k's, len(training_sizes)
    """
    test_scores_list = []
    test_scores_std = []
    k_vals = [k_range] if type(k_range) == int else range(k_range[0], k_range[1]+1, 2)
    with open(f'{kernel_ID}_n{n}_opt{optimize}.txt', 'w') as fhandle:
        for k in k_vals:
            X, y = get_X_y(file_name=f'n{n}/RandomSampling/n{n}_k{k}_randSamp_policy.mat')
            X_val, y_val, _, _ = get_subset_n(X,y,np.round(training_sizes[-1]/(1-1/n_folds)))
            if optimize:
                X_opt, y_opt, _, _ = get_subset_n(X,y,optimize_training_size)
                if composite:
                    print('Optimizing composite kernel...')
                    kernel = build_composite_kernel(X_opt,y_opt,n_layer,kernel_basis)
                else:
                    print('Optimizing kernel with BO...')
                    kernel,_,_ = optimize_kernel(kernel,X_opt,y_opt)
                print(f'Done! Optimized kernel is {kernel}')
            print(f'---Calculating learning curve for kernel = {kernel}; n = {n}; k = {k}---')
            test_scores, training_sizes = get_learning_curve(kernel, X_val, y_val, n_folds, training_sizes)
            mean_test_scores = list(np.mean(test_scores,axis=1))
            std_test_scores = list(np.std(test_scores,axis=1))
            test_scores_list.append(mean_test_scores)
            test_scores_std.append(std_test_scores)
            write_row_to_file(fhandle,mean_test_scores)
            write_row_to_file(fhandle,std_test_scores)
    return test_scores_list, test_scores_std, training_sizes

def write_row_to_file(fhandle, result_list, fmt='%.6f'):
    """
    Write a numerical list to a row in file
    Args:
        fhandle: TextIO, handle of an open file
        result_list: list, list containing numerical results to write to file
        fmt: format string
    """
    formatted_list = [fmt % result for result in result_list]
    fhandle.write(', '.join(formatted_list) + '\n')

def get_learning_curve(kernel, X ,y, n_folds, training_sizes):
    """
    Get the learning curve of the given kernel, for the validation set (X,y)
    Args:
        kernel: Kernel, kernel to be evaluated
        X: np.array, design matrix of size (dim x n_training_points)
        y: np.array, target training vector of size (n_training_points,)
        n_folds: int, number of fold for cross-validation
        training_sizes: list[int], list of step training sizes for the learning curve

    Returns:
        train_sizes_abs, test_scores
    """
    clf = SVC(kernel=kernel)
    train_sizes, _, test_scores = learning_curve(clf,X,y,train_sizes=training_sizes,cv=n_folds, shuffle=True,
                                                           scoring='accuracy',n_jobs=-1,verbose=2)
    return test_scores, train_sizes

def get_roc_curve(kernel,n,k,optimize=False):
    """ Get roc curve for (n,k)-forrelation dataset in policy space with thres = 0.5"""
    data_dir = data_dir
    X, y = get_X_y(data_dir+f"n{n}/n{n}_k{k}_randSamp_policy.mat")
    X_train, y_train, X_test, y_test = get_training_test_data(X, y,10000,5000)
    if optimize:
        X_opt, y_opt, _, _ = get_subset_n(X_train, y_train, 5000)
        kernel, _, _ = optimize_kernel(kernel, X_opt, y_opt)
    clf = SVC(kernel=kernel,probability=True)
    clf.fit(X_train,y_train)
    prob_pos = clf.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test,prob_pos,pos_label=1)
    return fpr, tpr, thresholds

def get_roc_curves_increasing_k(kernel,optimize=False):
    """ Get roc curves for increasing values of k. NOTE: hardcode for n_vals and k_vals"""
    n_vals = range(3,7,1)
    k_vals = range(9,22,2)
    roc_data_storage = np.zeros((len(n_vals),len(k_vals)),dtype=tuple)
    for i in range(len(n_vals)):
        for j in range(len(k_vals)):
            print(f"Getting ROC curve for n={n_vals[i]} ; k={k_vals[j]}...")
            fpr, tpr, _ = get_roc_curve(kernel,n_vals[i],k_vals[j],optimize)
            print("Done!")
            roc_data_storage[i][j] = (fpr, tpr)
            np.save("results/roc_data_storage.npy",roc_data_storage)
    return roc_data_storage

def optimize_kernel(kernel, X, y):
    """
    Optimize the hyperparameters of a kernel on a validation set
    Args:
        kernel: Kernel, kernel to be optimized
        X: np.ndarray, validation set of size (n_validation_points x dim)
        y: np.ndarray, target validation vector of size (n_validation_points,)

    Returns:
        optimized_kernel: Kernel, kernel with optimized hyperparameters
        BIC: float, Bayesian Information Criterion of the optimized kernel
        log_LLH: float, log likelihood of the optimized kernel
    """
    # TODO: upgrade to optimize classifier, which return optimized kernel and C (not a priority)
    print(f"Optimizing Kernel: {kernel}")
    kernel = kernel.clone_with_theta(kernel.theta)
    search_space = get_search_space(kernel)
    @use_named_args(dimensions=search_space)
    def cost_func(**params):
        kernel.set_params(**params)
        clf = SVC(kernel=kernel,probability=True)
        return -np.average(cross_val_score(clf,X,y,scoring='neg_log_loss',cv=5,n_jobs=-1))

    # implement optimizer to minimize cost funtion (gbrt_minimize or gp_minimize)
    # Bayesian Optimization
    xi = 0.01
    # init_args = list(np.exp(kernel.theta))
    bo_results = gp_minimize(cost_func, dimensions=search_space,
                            #  callback= callback_BO,
                             n_calls=10, n_initial_points=5,
                             acq_func='EI', acq_optimizer='lbfgs',
                             n_restarts_optimizer=3,
                             xi=xi, random_state=2508, n_jobs=-1)
    optimized_kernel = push_params(kernel,bo_results.x)
    Log_loss = bo_results.fun
    BIC = -2*(-Log_loss) + len(search_space) * np.log(np.shape(X)[0] / 3)
    return optimized_kernel, BIC, Log_loss

def generate_possible_kernels(base_kernel,kernel_basis=None):
    """
    Generate a list of possible kernels using the base kernel
    Args:
        base_kernel: Kernel, base kernel
        kernel_basis: list[Kernel], list containing basis kernels

    Returns:
        possible_kernels: list[Kernel]
    """
    #TODO: change generate possible kernels to only add CONST when needed
    if kernel_basis is None:
        kernel_basis = DEFAULT_BASIS
    possible_kernels = []
    for kernel in kernel_basis:
        possible_kernels.append(base_kernel*kernel)
        possible_kernels.append(base_kernel + CONST*kernel)
    return possible_kernels

def callback_BO(res):
    """
    Args:
        res [OptimizeResult]: scipy object containing the current status of BO
    """
    last_location = res.x_iters[-1]
    cost_min = res.fun
    print(f'Last location: {last_location} ; Current min: {cost_min}')

def get_search_space(kernel):
    """
    Get hyperparameter search space of kernel
    Args:
        kernel: Kernel, kernel with hyperparameters to extract search space
    Returns:
        list(Space): list of dimensions for hyperparameter search
    """
    space = []
    for hyperparam in kernel.hyperparameters:
        space.append(Real(name=hyperparam[0], low=hyperparam[2][0,0], high=hyperparam[2][0,1],
                          prior="log-uniform", base=np.e))
    return space

def push_params(kernel, params):
    """
    Load learned parameters into a kernel
    Args:
        kernel: Kernel, kernel to load learned parameters
        params: list, list of hyperparameter values
    Returns:
        kernel with learned hyperparameters
    """
    params_dict = {}
    hyperparams_list = kernel.hyperparameters
    for i in range(len(params)):
        params_dict[hyperparams_list[i][0]] = params[i]
    return kernel.set_params(**params_dict)

def clone(kernel):
    return kernel.clone_with_theta(kernel.theta)





