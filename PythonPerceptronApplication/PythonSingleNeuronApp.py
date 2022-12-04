
# Python program to implement a single neuron neural network
# to separate dots to Positive and Negative on two-dimensional plane relatively to X-axis
 
# Class to create a neural network with single neuron
class NeuralNetwork():
    MARGIN_OF_ERROR = 0.0001;
    train_dataset_errors = []

    positive_dots_array = [];
    negative_dots_array = [];
    x_synapse_signal = 0;
    y_synapse_signal = 0;
    x_synapse_weight = 0.5;
    y_synapse_weight = 0.5;

    def clear_dot_arrays(self):
        self.positive_dots_array = [];
        self.negative_dots_array = [];
     
    def get_synapse_weights(self):
        return [self.x_synapse_weight, self.y_synapse_weight]

    def set_synapse_weights(self, x_synapse_weight, y_synapse_weight):
        self.x_synapse_weight = x_synapse_weight;
        self.y_synapse_weight = y_synapse_weight;

    def get_activation_criterion(self, synapse_signal):
        return 1 if synapse_signal >= 0 else 0
 
    # activation_function
    def check_is_activation_condition(self, x_synapse_signal, y_synapse_signal):
        activation_condition = self.get_activation_criterion(x_synapse_signal) * self.x_synapse_weight + self.get_activation_criterion(y_synapse_signal) * self.y_synapse_weight;

        return activation_condition == 1

    def separate_dots(self, dotsArray):
        for dot in dotsArray:
            self.positive_dots_array.append(dot) if self.check_is_activation_condition(*dot) else self.negative_dots_array.append(dot)

        return [self.positive_dots_array, self.negative_dots_array]

 
    # training the neural network
    def train(self, train_y_synapse_signal_input, train_activation_criterion_output, num_of_train_iterations):
        # Since only two classes are needed and dots are separated relatively to X-axis, it's enough to use only one Y-synapse signal

        train_dataset_length = len(train_y_synapse_signal_input)

        # Check that training sets hahe appropriate length
        if train_dataset_length != len(train_activation_criterion_output) :
           print('input and output training arrays must have the same length');
           return
                                 
        # Number of iterations we want to perform for this set of input
        for iteration in range(num_of_train_iterations):
            for idx, yCoordinate in enumerate(train_y_synapse_signal_input):
                output = self.get_activation_criterion(yCoordinate) * self.y_synapse_weight
                print ('output', output)
                print ('train_activation_criterion_output', train_activation_criterion_output[idx])
 
                # Calculate the error in the output.
                error = train_activation_criterion_output[idx] - output
                print ('error', error)
 
                adjustment = 0.5 * error ** 2
                print ('adjustment', adjustment)
                              
                # Adjust the weight matrix
                new_y_synapse_weight = self.y_synapse_weight + adjustment
                new_x_synapse_weight = 1 - new_y_synapse_weight
                self.set_synapse_weights(new_x_synapse_weight, new_y_synapse_weight)

                # check, whether required precision has reached for all coordinates in train set
                if error < self.MARGIN_OF_ERROR :
                    self.train_dataset_errors.append(error)
                    if len(self.train_dataset_errors) >= train_dataset_length : 
                        print('required precision ', self.MARGIN_OF_ERROR,  'has reached, current iteration: ', iteration)
                        self.set_synapse_weights(0, 1)
                        return

                if idx + 1 == train_dataset_length : 
                    self.train_dataset_errors.clear()
 
# Driver Code
if __name__ == "__main__":
     
    neural_network = NeuralNetwork()

    artray_to_separate = [[-10, 3], [-1, -2], [-1, 1], [1, -1], [1, 0], [0, 0], [0, 1], [-1, -1], [1, 1], [2, 2], [2, -1]]

    separated_arrays_with_untrained_neuron = neural_network.separate_dots(artray_to_separate);
    
    print ('Initial array of dots to to separate')
    print (artray_to_separate);
    print ()
    print ('Initial X coordinate synapse weight and Y coordinate synapse weight at the start of training')
    print (neural_network.get_synapse_weights())
    print ()
    print ('Untrained neural network positive Y-coordinate dots')
    print (separated_arrays_with_untrained_neuron[0])
    print ()
    print ('Untrained neural network negative Y-coordinate dots')
    print (separated_arrays_with_untrained_neuron[1])
    print ()
 
    train_y_synapse_signal_input = [3, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    train_activation_criterion_output = [1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
 
    neural_network.train(train_y_synapse_signal_input, train_activation_criterion_output, 10000)
 
    print ('New weights after training')
    print (neural_network.get_synapse_weights())
    print ()
 
    # Test the trained neural network
    neural_network.clear_dot_arrays()

    separated_arrays_with_trained_neuron = neural_network.separate_dots(artray_to_separate);

    print ('Trained neural network positive Y-coordinate dots')
    print (separated_arrays_with_trained_neuron[0])
    print ()
    print ('Trained neural network negative Y-coordinate dots')
    print (separated_arrays_with_trained_neuron[1])
    print ()
