# Neural-Network-From-Scratch
Build and train a fully custom neural network (no TensorFlow / PyTorch) on the cereals dataset to predict a target value from nutritional features, and evaluate prediction accuracy.

**What the code does:**
1. Define a Neural_Network class with:
   - Arbitrary layer sizes (inputs, hidden layers list, outputs)
   - Random weight initialization
   - Feedforward pass using sigmoid activation
   - Backpropagation with sigmoid derivative and mean squared error (MSE)
   - Momentum term for weight updates
   - Training function with gradient descent + momentum
   - Prediction function with simple error-based accuracy
   - Normalization helper
   - Plotting function for actual vs predicted outputs

**2. Load and prepare data:**
- Read cereals.csv with pandas
   - Convert to NumPy array
   - Split into:
     X = all columns except last (inputs)
     y = last column (output, reshaped to column vector)
   - Normalize y
   - Concatenate X and y back into a single data array

**3. Train/test split:**
- TRAINING_PERCENTAGE = 75 (75% training, 25% testing)
   - data_training = first 75% of rows
   - data_testing  = remaining 25%
   - trainingSet  = first 5 columns (features) of training data
   - testing_set  = first 5 columns (features) of test data
   - trainingTargetSet = 6th column of training data (target)
   - testing_output_set = 6th column of test data

**4. Neural network configuration:**
- Input nodes: 5
   - Hidden layer: [3]  (one hidden layer with 3 neurons)
   - Output nodes: 1
   - n_epochs = 10
   - LearningRate = 0.9
   - Momentum    = 0.4
   - PRED_ERROR  = 0.2 (threshold for counting a prediction as correct)

**5. Training loop:**
- Optionally shuffle training data n_datashuffle times (here 1 time)
   - For each epoch:
       * Feedforward on each training sample
       * Compute error (target - output)
       * Backpropagate error to compute derivatives
       * Compute momentum term
       * Update weights with learning rate + momentum
       * Accumulate and print average MSE if verbose=True

**6. Prediction and evaluation:**
- Use mlp.predict(testing_set, testing_output_set)
   - For each target:
       * Feedforward to get predicted output
       * Compute MSE between target and prediction
       * Count as correct if error <= PRED_ERROR
   - Return predictionTable = [correct_count, incorrect_count]
   - Compute accuracy = correct / (correct + incorrect) * 100
   - Print accuracy in percent
   - Plot actual vs predicted outputs

**Dependencies:**
pip install numpy pandas matplotlib

**Output:**
- Printed shapes of input/output and concatenated data
- Printed training and testing shapes
- Printed MSE per epoch (if verbose=True in train)
- Final predicted table [correct, incorrect]
- Final accuracy percentage
- A Matplotlib plot comparing actual vs predicted outputs

**Dataset**
- cereals.csv
- wisc_bc_data.csv
