from LSTM import LSTM

class Init:

    def __init__(self):            

        # Init default epoch and l_r
        self.epoch = 0
        self.learning_rate = 0

        # Load the book and convert it into lowercase
        self.data = open('book.txt').read().lower()

        self.characters = set(self.data)
        self.size = len(self.characters)

        self.char_to_id = {word: index for index, word in enumerate(self.characters)}
        self.id_to_char = {index: word for index, word in enumerate(self.characters)}

    def enter_parameters(self):
        print('+++ Enter network parameters +++')

        # Store the epoch and the learning rate
        self.epochs = int(input('Enter the number of epochs: '))
        self.learning_rate = float(input('Enter the learning rate [0.0001 - 0.1]: '))

    def run_model(self):            
        # Create the LSTM network
        model = LSTM(self.char_to_id, self.id_to_char, self.size, \
            epochs = self.epochs, lr = self.learning_rate)

        # Train the model
        losses = model.train_model(data) 


init = Init()

init.enter_parameters()

init.run_model()
