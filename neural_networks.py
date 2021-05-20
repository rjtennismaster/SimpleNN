import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

#sigmoid function to normalize predictions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#derivative of sigmoid
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

#generates a list of 3 random numbers between -2 and 2
def random_number_list():
    num_list = []
    
    for _ in range(3):
        num_list.append(random.uniform(-2, 2))
    
    return num_list

#2b
def mean_squared_error(vectors, bias_and_weights, classses, png_file_name):
    df = pd.read_excel(vectors)
    df.columns = ['petal_length', 'petal_width', 'species']
    input_df = df.drop(columns = ['species'])

    #input dataset
    inputs = input_df.to_numpy()

    species_list = df['species'].tolist()

    #actual outputs
    actual_outputs = np.array(species_list).T

    #acknowledge passed weight parameters
    synaptic_weights = np.array([bias_and_weights[1], bias_and_weights[2]]).T

    #normalize product of inputs * synaptic weights
    predictions = sigmoid(np.dot(inputs, synaptic_weights) + bias_and_weights[0])
    
    #plot the predictions with their respective colors. 0 = veriscolor(blue) and 1 = virginica(red)
    plt.grid()

    for i in range(len(inputs)):
        color = 'b'
        if predictions[i] >= 0.5:
            color = 'r'
        plt.scatter([inputs[i][0]], [inputs[i][1]], color = color)

    plt.title("Predictions")
    plt.xlabel("pedal length")
    plt.ylabel('pedal width')
    plt.savefig(png_file_name)


    #calculate mean squared error
    squared_errors = np.square(predictions - actual_outputs)

    sum_of_squared_errors = np.sum(squared_errors)

    result = sum_of_squared_errors / squared_errors.size
    print('mean squared error for this dataset:')
    print(result)
    return result

#2e. Plots difference between decision boundaries of 999th and 1000th iteration
def summed_gradients(vectors, bias_and_weights):
    df = pd.read_excel(vectors)
    df.columns = ['petal_length', 'petal_width', 'species']

    #dataset
    data = df.to_numpy()
    print('Dataset:')
    print(data)

    bias = bias_and_weights[0]
    w1 = bias_and_weights[1]
    w2 = bias_and_weights[2]

    iterations = 1000
    learning_rate = 0.1
    #gradient sum storage. index 0 is bias, 1, is w1, 2 is w2
    gradients = [0, 0, 0]
     
    for iteration in range(iterations):
        if iteration == 998:
            #update parameters (take a step)
            bias = bias - (gradients[0] * learning_rate)
            w1 = w1 - (gradients[1] * learning_rate)
            w2 = w2 - (gradients[2] * learning_rate)
            gradients = [0, 0, 0]

            for j in range(len(data)):
                point = data[j]
                z = bias + point[0] * w1 + point[1] * w2
                prediction = sigmoid(z)
                actual = point[2]

                #squared_error = np.square(prediction - actual)

                #take derivative of obj func to get respective gradients, and add them to respective gradient sums
                derror_dprediction = 2 * (prediction - actual)
                dprediction_dz = sigmoid_prime(z)

                dz_dbias = 1
                dz_dw1 = point[1]
                dz_dw2 = point[2]

                derror_dz = derror_dprediction * dprediction_dz

                #gradient of bias
                derror_dbias = derror_dz * dz_dbias

                #gradient of w1
                derror_dw1 = derror_dz * dz_dw1

                #gradient of w2
                derror_dw2 = derror_dz * dz_dw2

                #update gradient sums
                gradients[0] += derror_dbias
                gradients[1] += derror_dw1
                gradients[2] += derror_dw2

            mean_squared_error(vectors, [bias, w1, w2], [0, 1], 'decision1.png')
        elif iteration == 999:
            #update parameters (take a step)
            bias = bias - (gradients[0] * learning_rate)
            w1 = w1 - (gradients[1] * learning_rate)
            w2 = w2 - (gradients[2] * learning_rate)
            gradients = [0, 0, 0]

            for j in range(len(data)):
                point = data[j]
                z = bias + point[0] * w1 + point[1] * w2
                prediction = sigmoid(z)
                actual = point[2]

                #squared_error = np.square(prediction - actual)

                #take derivative of obj func to get respective gradients, and add them to respective gradient sums
                derror_dprediction = 2 * (prediction - actual)
                dprediction_dz = sigmoid_prime(z)

                dz_dbias = 1
                dz_dw1 = point[1]
                dz_dw2 = point[2]

                derror_dz = derror_dprediction * dprediction_dz

                #gradient of bias
                derror_dbias = derror_dz * dz_dbias

                #gradient of w1
                derror_dw1 = derror_dz * dz_dw1

                #gradient of w2
                derror_dw2 = derror_dz * dz_dw2

                #update gradient sums
                gradients[0] += derror_dbias
                gradients[1] += derror_dw1
                gradients[2] += derror_dw2

            mean_squared_error(vectors, [bias, w1, w2], [0, 1], 'decision2.png') 
        else:
            #update parameters (take a step)
            bias = bias - (gradients[0] * learning_rate)
            w1 = w1 - (gradients[1] * learning_rate)
            w2 = w2 - (gradients[2] * learning_rate)
            gradients = [0, 0, 0]

            for j in range(len(data)):
                point = data[j]
                z = bias + point[0] * w1 + point[1] * w2
                prediction = sigmoid(z)
                actual = point[2]

                #squared_error = np.square(prediction - actual)

                #take derivative of obj func to get respective gradients, and add them to respective gradient sums
                derror_dprediction = 2 * (prediction - actual)
                dprediction_dz = sigmoid_prime(z)

                dz_dbias = 1
                dz_dw1 = point[1]
                dz_dw2 = point[2]

                derror_dz = derror_dprediction * dprediction_dz

                #gradient of bias
                derror_dbias = derror_dz * dz_dbias

                #gradient of w1
                derror_dw1 = derror_dz * dz_dw1

                #gradient of w2
                derror_dw2 = derror_dz * dz_dw2

                #update gradient sums
                gradients[0] += derror_dbias
                gradients[1] += derror_dw1
                gradients[2] += derror_dw2

    print('bias: ')
    print(bias)
    print('w1: ')
    print(w1)
    print('w2: ')
    print(w2)

#3a & b. Trains the NN through gradient descent to reduce the error as much as possible (i.e. optimize decision boundary). Plots: Ending decision boundary, learning curve
def train_gradient_descent(vectors, bias_and_weights):
    df = pd.read_excel(vectors)
    df.columns = ['petal_length', 'petal_width', 'species']

    #convert data set to numpy array
    data = df.to_numpy()
    print('Dataset:')
    print(data)

    bias = bias_and_weights[0]
    w1 = bias_and_weights[1]
    w2 = bias_and_weights[2]

    #number of iterations, 10000 is standard
    iterations = 10000
    learning_rate = 0.1

    #keep track of squared error at each iteration
    squared_errors = np.array([])

    #keep track of mean squared error so we can plot it eventually
    mean_squared_errors = np.array([])

    for i in range(iterations):
        #use a random point in the dataset
        random_num = np.random.randint(len(data))
        point = data[random_num]

        #decision boundary equation
        z = bias + point[0] * w1 + point[1] * w2
        prediction = sigmoid(z)
        actual = point[2]
        squared_error = np.square(prediction - actual)

        squared_errors = np.append(squared_errors, squared_error) 

        mean_squared_errors = np.append(mean_squared_errors, (np.sum(squared_errors) / squared_errors.size))

        #if the mean squared error is less than 0.001, we can stop and print out the values of the parameters
        if mean_squared_errors[i] < 0.001:
            print("we're ending early! ending parameters:")
            print(' ')
            print('bias: ')
            print(bias)
            print('w1: ')
            print(w1)
            print('w2: ')
            print(w2)

            return

        #print the parameters whe we reach halfway through our iterations
        if i == iterations / 2:    
            print('parameters halfway through: ')
            print('')
            print('bias: ')
            print(bias)
            print('w1: ')
            print(w1)
            print('w2: ')
            print(w2)

        #take the derivative of the objective func to get the gradients and update the parameters
        derror_dprediction = 2 * (prediction - actual)
        dprediction_dz = sigmoid_prime(z)

        dz_dbias = 1
        dz_dw1 = point[1]
        dz_dw2 = point[2]

        derror_dz = derror_dprediction * dprediction_dz

        #gradient of bias
        derror_dbias = derror_dz * dz_dbias

        #gradient of w1
        derror_dw1 = derror_dz * dz_dw1

        #gradient of w2
        derror_dw2 = derror_dz * dz_dw2

        #figure out step sizes (how much we'll change each parameter)
        bias_step_size = derror_dbias * learning_rate
        w1_step_size = derror_dw1 * learning_rate
        w2_step_size = derror_dw2 * learning_rate

        #finally, change the weights based on the step size
        bias -= bias_step_size
        w1 -= w1_step_size
        w2 -= w2_step_size

        #print the values of the parameters if on last iteration
        if i == (iterations - 1):
            print('ending parameters:')
            print(' ')
            print('bias: ')
            print(bias)
            print('w1: ')
            print(w1)
            print('w2: ')
            print(w2)

    #plot the learning curve
    plt.title('Learning Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.plot(mean_squared_errors)
    plt.savefig('learningCurve.png')

    #plot the decision boundary and return mean squared error of whole data set
    #mean_squared_error(vectors, [bias, w1, w2], [0, 1], 'ending decision.png') <--wanted to call this, but the plotting is acting wacky so I'll just call MSE with the printed weights




#3c
#starting decision boundary:
#random_starting_parameters = random_number_list()
#print('random starting parameters:')
#print(random_starting_parameters)
#mean_squared_error('irisdata.xlsx', random_starting_parameters, [0, 1], 'start_decision.png')
#train gradient func will let us know the parameters at halfway through & at the end so we can plug them in and plot the decision boundary
#train_gradient_descent('irisdata.xlsx', random_starting_parameters)

#now we know midway and ending parameters, so we can plug them into plotting func to visualize
#mean_squared_error('irisdata.xlsx', [-7.599015327148707, -4.490375476905719, 18.38132944550192] , [0, 1], 'midway_decision.png')
#mean_squared_error('irisdata.xlsx', [-11.468725816758589, -6.942480435029876, 27.791309218983198] , [0, 1], 'end_decision.png')



#some more arguments we tried to test the program

#train_gradient_descent('irisdata.xlsx', [-.000356765486754765, -0.00004500000009, 0.0003398])
#mean_squared_error('irisdata.xlsx', [-11.924946642784034, -6.755993497684189, 27.633176745966946], [0, 1], 'ending_decision.png')

#summed_gradients('irisdata.xlsx', [-.000356765486754765, -0.00004500000009, 0.0003398])

#mean_squared_error('irisdata.xlsx', [0, 2.4, 2.7], [0, 1])
#mean_squared_error('irisdata.xlsx', [-.000356765486754765, -0.00004500000009, 0.0003398], [0, 1], 'decision1.png')
