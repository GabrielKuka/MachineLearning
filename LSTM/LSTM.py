import numpy as np
from random import uniform
from .Xavier import Xavier
from .Helper import Helper
from .Adam import Adam

class LSTM:

    def __init__(self, char_to_id, id_to_char, voc_size, hd_units=80, recurrences=32,
                 epochs=10, lr=0.001):

        # ~~~~~~ Mapping ~~~~~~
        self.char_to_id = char_to_id  # Map each char to an id
        self.id_to_char = id_to_char  # Map each id to a character
        
        # Small constant to avoid division by 0
        self.EPS = 1e-8

        # Init Adam Optimizer Helper
        adam = Adam(self.EPS)

        self.voc_size = voc_size  # no. of unique characters in the training data
        self.hd_units = hd_units  # no. of units in the hidden layer
        self.epochs = epochs
        self.lr = lr  # learning rate
        self.recurrences = recurrences  # no. of time steps

        # Save the weights and biases of the network in a dictionary
        self.params = {}

        # Init weights and biases
        self.init_weights_and_biases()

        # Initialise gradients and Adam parameters
        self.adam_params = {}
        self.gradients = {}

        # Init gradients
        self.init_gradients()

        # The loss function formula
        self.smooth_loss = -np.log(1.0 / self.voc_size) * self.recurrences


    def init_weights_and_biases(self):

        # Initialize Xavier 
        xavier = Xavier(self.voc_size, self.hd_units)

        # Weights and biases for the Forget Gate
        self.params["Wf"] = xavier.get_random_weights() 
        self.params["bf"] = xavier.get_forget_gate_biases() 

        # Weights and biases for the Input Gate
        self.params["Wi"] = xavier.get_random_weights() 
        self.params["bi"] = xavier.get_random_biases()

        # Weights and biases for the Output Gate
        self.params["Wo"] = xavier.get_random_weights() 
        self.params["bo"] = xavier.get_random_biases() 

        # Weights and biases for the Cell State
        self.params["Wc"] = xavier.get_random_weights() 
        self.params["bc"] = xavier.get_random_biases()

        # Weights and biases for the output
        self.params["Wv"] = xavier.get_output_weights() 
        self.params["bv"] = xavier.get_output_biases() 


    def init_gradients(self):
        """ For each weight and bias, assign gradients with 0 """

        for key in self.params:
            self.gradients["d" + key] = np.zeros_like(self.params[key])
            self.adam_params["m" + key] = np.zeros_like(self.params[key])
            self.adam_params["v" + key] = np.zeros_like(self.params[key])


    def reset_gradients(self):
        """ Before each propagation, reset gradients to 0 """

        for key in self.gradients:
            self.gradients[key].fill(0)


    def minimize_gradients(self):
        """
        Limits the magnitude of gradients to avoid exploding gradients
        """
        for key in self.gradients:
            np.clip(self.gradients[key], -5, 5, out=self.gradients[key])


    def update_parameters(self, time_step):
        """
        Adam optimizer 
        """
        # For every weight and bias in the neural network, update the parameters
        for key in self.params:
            self.adam_params["m" + key] =  \
                adam.update_m(self.adam_params['m'+key], self.gradients['d'+key])

            self.adam_params['v'+key] = \
                adam.update_v(self.adam_params['v'+key], self.gradients['d'+key])

            # Apply bias correction for better results
            m_corrected, v_corrected = \
                adam.bias_correction(self.adam_params['m'+key], \
                    self.adam_params['v'+key], time_step)

            # Update the weight
            self.params[key] -= self.lr * m_corrected / (np.sqrt(v_corrected)
                                                          + self.EPS)


    def forward_propagate(self, x, h_prev, c_prev):
        """ Forward propagation for one time step """

        # Add the input with the previous hidden state
        z = np.row_stack((h_prev, x))

        # Calculate Forget Gate
        forget_gate = Helper.sigmoid(np.dot(self.params["Wf"], z) + self.params["bf"])

        # Calculate Input gate
        input_gate = Helper.sigmoid(np.dot(self.params["Wi"], z) + self.params["bi"])
        c_bar = Helper.tanh(np.dot(self.params["Wc"], z) + self.params["bc"])

        # Calculate Output Gate
        output_gate = Helper.sigmoid(np.dot(self.params["Wo"], z) + self.params["bo"])

        # Calculate the new Cell State
        cell_state = forget_gate * c_prev + input_gate * c_bar

        # Calculate the new hidden state
        hidden_state = output_gate * Helper.tanh(cell_state)

        # Calculate the output
        v = np.dot(self.params["Wv"], hidden_state) + self.params["bv"]

        # Apply softmax
        y_hat = Helper.softmax(v)

        return y_hat, v, hidden_state, output_gate, cell_state, \
            c_bar, input_gate, forget_gate, z


    def back_propagate(self, y, y_hat, dh_next, dc_next, c_prev, z, fg, ig,
                       c_bar, c, og, h):
        """
            Backpropagate in one recurrence
        """

        # Get the error of the network
        error = np.copy(y_hat)
        error[y] -= 1  # yhat - y

        # ~~~~~~~~ Derivatives of the Error with respect to the states ~~~~~~~~ #

        # Derivative of the error with respect to the hidden state:
        """
            dE/dh_t = [dE/dy_hat] * [dy_hat/d(v*h_t)] * [d(v*h_t)/dh_t]
            dE/dh_t = v * (y_hat - y)
        """
        dh_t = np.dot(self.params["Wv"].T, error)
        dh_t += dh_next

        # Derivative of the error with respect to the cell state
        """ 
            dE/dc_t = [dE/dy_hat] * [dy_hat/dh_t] * [dh_t/dc_t]
                    = [dE/dh_t] * [dh_t/dc_t]
                    = [(y_hat - y) * v] * o_t * [1 - tanh(c_t)^2]
                    = dh_t * o_t * [1 - tanh(c_t)^2]  
        """
        dc_t = dh * og * (1 - Helper.tanh(c) ** 2)
        dc_t += dc_next


        # ~~~~~~~~ Derivatives of the error with respect to the gates ~~~~~~~~ #

        # Derivative of the error with respect to the output gate
        """
            dE/do_t = [dE/dh_t]*[dh_t/do_t]
                    = [dE/dh_t] * tanh(c_t)
        """
        do_t = dh * Helper.tanh(c)

        # Derivative of the error with respect to the input gate
        """
            dE/di_t = [dE/dh_t]*[dh_t/dc_t]*[dc_t/di_t]
                    = [dE/dc_t]*[dc_t/di_t]
                    = [dE/dc_t] * c_bar
        """
        di_t = dc * c_bar

        # Derivative of the error with respect to the candidate gate 
        """
            dE/dc_bar = [dE/dy_hat]*[dy_hat/dh_t]*[dh_t/dc_t]*[dc_t/dc_bar]
                      = [dE/dc_t]*[d_ct/dc_bar]
                      = [dE/dc_t]*i_t
        """
        dc_bar = dc * ig

        # Derivative of the error with respect to the forget gate
        """
            dE/df_t = [dE/hd_t]*[dh_t/dc_t]*[dc_t/df_t]
                    = [dE/c_t] * [dc_t/df_t]
                    = [dE/c_t] * c_prev
        """
        df_t = dc * c_prev


        # ~~~~~~~~ Derivatives of the error with respect to the W and b of the gates ~~~~~~~~ #

        # Derivative of the Error with respect to v
        """
            dE/dv = sum(dE_i/dv)
                  = sum( dE_i/dy_hat * dy_hat/dz_t * dz_t/dv )
                  = sum( (y_hat - y_i) * h_i )
        """
        self.gradients["dWv"] += np.dot(error, h.T)
        self.gradients["dbv"] += error

        # Derivative of the Error with respect to the W and b of output gate
        """
            dE/dwo = sum( [dE/do_t] * [do_t/dwo] )
                   = sum( dE/do_t * sig( wo * h_t-1 * bo ) * (1 - sig( wo * h_t-1 * bo )) * h_t-1 )
                   = sum( dE/do_t * o_t * (1 - o_t) * h_t-1 )
        """
        dp_o = do * og * (1 - og)
        self.gradients["dWo"] += np.dot(dp_o, z.T)
        self.gradients["dbo"] += dp_o

        # Derivative of the Error with respect to the W and b of input gate
        """
            dL/dwi = sum( [dE/di_t] * [di_t/dwi] )
                   = sum( [dE/di_t] * [sig( wi * h_t-1 + bi ) * (1 - sig( wi * h_t-1 + bi )) * h_t-1] )
                   = sum( [dE/di_t] * i_t * (1 - i_t) * h_t-1 )
        """
        dp_i = di * ig * (1 - ig)
        self.gradients["dWi"] += np.dot(dp_i, z.T)
        self.gradients["dbi"] += dp_i

        # Derivative of the Error with respect to the W and b of candidate gate
        """
            dE/dwc_bar = sum( [dE/dc_bar * [dc_bar/dwc_bar] )
                       = sum( [dE/dc_bar] * ( 1 - tanh(wc_bar * h_t-1 + bc_bar)^2 ) * h_t-1 )
                       = sum( [dE/dc_bar] * ( 1 - c_bar^2 ) * h_t-1 )
        """
        dp_c = dc_bar * (1 - c_bar ** 2)
        self.gradients["dWc"] += np.dot(dp_c, z.T)
        self.gradients["dbc"] += dp_c

        # Derivative of the Error with respect to the W and b of forget gate
        """
            dE/dwf = sum( [dE/df_t] * [df_t/dwf] )
                   = sum( [dE/df_t] * f_t * (1 - f_t) * h_t-1 )
        """
        dp_f = df * fg * (1 - fg)
        self.gradients["dWf"] += np.dot(dp_f, z.T)
        self.gradients["dbf"] += dp_f

        dz = (np.dot(self.params["Wf"].T, dp_f)
              + np.dot(self.params["Wi"].T, dp_i)
              + np.dot(self.params["Wc"].T, dp_c)
              + np.dot(self.params["Wo"].T, dp_o))

        dh_prev = dz[:self.hd_units, :]
        dc_prev = fg * dc

        return dh_prev, dc_prev


    def forward_backward(self, x_batch, y_batch, h_prev, c_prev):
        """ Forward prop, backward prop for one mini-batch """

        # Lists to store gates and states values
        fg, ig, c_bar, c, og = {}, {}, {}, {}, {}

        # Lists to store output values
        y_hat, v, h = {}, {}, {}

        # Lists to store the inputs
        x, z = {}, {}

        h[-1] = h_prev
        c[-1] = c_prev

        loss = 0

        # For each recurrence in the mini-batch, forward propagate and calculate the loss
        for t in range(self.recurrences):
            x[t] = np.zeros((self.voc_size, 1))
            x[t][x_batch[t]] = 1

            y_hat[t], v[t], h[t], og[t], c[t], c_bar[t], ig[t], fg[t], z[t] = \
                self.forward_propagate(x[t], h[t - 1], c[t - 1])

            loss += -np.log(y_hat[t][y_batch[t], 0])

        # Reset gradients before backpropagating
        self.reset_gradients()

        dh_next = np.zeros_like(h[0])
        dc_next = np.zeros_like(c[0])

        # Back propagate for the current mini-batch
        for t in reversed(range(self.recurrences)):
            dh_next, dc_next = self.back_propagate(y_batch[t], y_hat[t], dh_next,
                                                  dc_next, c[t - 1], z[t], \
                                                   fg[t], ig[t], \
                                                  c_bar[t], c[t], og[t], h[t])

        return loss, h[self.recurrences - 1], c[self.recurrences - 1]


    def test_model(self, h_prev, c_prev, sample_size):
        """ Output a sequence from the model """

        x = np.zeros((self.voc_size, 1))
        hs = h_prev
        cs = c_prev
        result = ""

        for t in range(sample_size):
            y_hat, _, hs, _, cs, _, _, _, _ = self.forward_propagate(x, hs, cs)

            # Pick a random index within the probability distribution
            index = np.random.choice(range(self.voc_size), p=y_hat.ravel())
            x = np.zeros((self.voc_size, 1))
            x[index] = 1

            # Add the character to the result
            char = self.id_to_char[index]
            result += char

        return result 


    def train_model(self, samples):
        """
        Main method of the LSTM class where training takes place
        """

        num_batches = len(samples) // self.recurrences

        # Trim input to have full sequences
        new_samples = samples[: num_batches * self.recurrences]

        training_size = len(new_samples) - self.recurrences

        # Store the losses produced for each batch
        losses = []

        for epoch in range(self.epochs):

            # Init empty cell and hidden state
            h_prev = np.zeros((self.hd_units, 1))
            c_prev = np.zeros((self.hd_units, 1))

            for a in range(0, training_size, self.recurrences):

                # Setup the batches
                input_batch = [self.char_to_id[ch] for ch in new_samples[a: a + self.recurrences]]
                output_batch = [self.char_to_id[ch] for ch in new_samples[a + 1: a + self.recurrences + 1]]

                # Forward and back propagate for this batch
                loss, h_prev, c_prev = self.forward_backward(input_batch, output_batch, h_prev, c_prev)

                # Smoothen loss value
                self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001
                losses.append(self.smooth_loss)

                self.minimize_gradients()

                # Calculate timestep and apply Adam optimizer
                time_step = epoch * self.epochs + a / self.recurrences + 1
                self.update_parameters(time_step)

                # Print out the results
                if a % 400000 == 0:
                    loss = round(self.smooth_loss, 3)
                    print('\n\nEpoch: {}, Loss: {}'.format(epoch, loss))
                    print('----------------------------------------------')

                    text = self.test_model(h_prev, c_prev, sample_size=450)

                    print('Result:')
                    print('~~~~~~~~~~~~~~~~~')
                    print('{}'.format(text))
                    print('~~~~~~~~~~~~~~~~~')

        return losses 
