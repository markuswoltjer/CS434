import numpy as np
import log_reg2 as lr

def format_data_csv(filename):
    """ Load in data and format it as basis for training or testing set """
    data = np.genfromtxt (filename, delimiter=",")
    Y = np.matrix(data[:,-1]).T
    X = np.matrix(data[:,0:-1])
    return(X,Y)


def format_data_csv2(filename):
    """ Load in data and format it as basis for training or testing set """
    data = np.genfromtxt (filename, delimiter=",")
    Y = data[:,-1]
    X = data[:,0:-1]
    return(X,Y)

def add_dummy(x):
    return np.concatenate((np.ones((len(x), 1)), x), 1)

def main():
    # 1
    (X_train, Y_train) = format_data_csv2('../data/usps-4-9-train.csv')
    (X_test, Y_test) = format_data_csv2('../data/usps-4-9-test.csv')
    for i in range(5,10):
        log_reg = lr.Log_Reg(add_dummy(X_train), Y_train, 'objective', 10**-i, 10)
        print("learning rate = 10^-" + str(i))
        print("Accuracy (predictions on right side of 0.5): "
              + str(lr.predict_accuracy(log_reg.w, add_dummy(X_train), Y_train)))
        print("Loss for training: " + str(log_reg.obj))
#    print (log_reg.w)
#    print(log_reg.get_prob(X_test[300]))

    # 2
    test_acc = []
    train_acc = []
    learn = 10**-9
    log_reg = lr.Log_Reg(add_dummy(X_train), Y_train, 'iteration', learn, 1)
    for i in range(100):
        test_acc.append(lr.predict_accuracy(log_reg.w, add_dummy(X_test), Y_test))
        train_acc.append(lr.predict_accuracy(log_reg.w, add_dummy(X_train), Y_train))
        log_reg.another_batch(add_dummy(X_train), Y_train, learn)

    # 3
    ''' The only change is the addition of the regularationzation
        of ||w||22.  The gradient of ||w||22 is 2w.
        See this page for proof
        https://math.stackexchange.com/questions/883016/gradient-of-l2-norm-squared/883024
        Of course gradient of L(w) + 1/2 ||w||22 would then be:
        [gradient of L(w)] + w

        Since the change in the objective function is deicrete in our case, the batch learning
        algorithm represents the gradient descent:
        w <- w - n * delta(L(w))
        as
        w <- w + n * d
        where d = -delta(L(w)) and is found thru the batch iteration
        To update this to account for L2 regularization, change the aforementioned line to:
        w <- w - n * delta(L(w) + 1/2 lambda * ||w||22)
        which is:
        w <- w + n * d - n * lambda * w
    '''

    # 4
    lams = [10**x for x in range(-3,4)]
    test_acc = []
    train_acc = []
    learn = 10 ** -9
    thresh = 100
    for lam in lams:
        log_reg = lr.Log_Reg(add_dummy(X_train), Y_train, 'iteration', learn, 50, lamb=lam)
        test_acc.append(lr.predict_accuracy(log_reg.w, add_dummy(X_test), Y_test))
        train_acc.append(lr.predict_accuracy(log_reg.w, add_dummy(X_train), Y_train))




if __name__ == "__main__":
    main()
