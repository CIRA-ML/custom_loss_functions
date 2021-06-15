# CIRA Guide to Custom Loss Functions for Neural Networks in Environmental Sciences

## This repository provides the code samples for the arXiv paper of the same title.

---
### MSE as custom loss function
```python
   def loss_MSE(y_true, y_pred):
      return tf.math.reduce_mean(tf.square(y_true - y_pred))
```

### Linking that loss function to the model
```python
   model.compile(optimizer=keras.optimizers.Adam(), loss=loss_MSE, metrics=['accuracy'])

```
---
###  Custom loss function to help with class imbalance
```python
   # Loss function with weights based on amplitude of y_true
   def my_MSE_weighted(y_true,y_pred):
      return K.mean(
          tf.multiply(
              tf.exp(tf.multiply(5.0, y_true)),
              tf.square(tf.subtract(y_pred, y_true))
          )
      )
```

---
### How to save and load a model that has a custom loss or metric

Consider a model that was trained with a custom loss function and saved in the usual manner.
```python
   model.save('K12.h5')
```
This example shows how to load that model, which was trained with the following custom loss function,
```python
    import tensorflow as tf
    import tensorflow.keras.backend as K
    
    def my_mean_squared_error_weighted_genexp(weight=(1.0,0.0,0.0)):
        def loss(y_true,y_pred):
            return K.mean(tf.multiply(
                tf.exp(tf.multiply(weight,tf.square(y_true))),
                tf.square(tf.subtract(y_pred,y_true))))
        return loss
```
and uses the metric,
```python
    def my_r_square_metric(y_true,y_pred):
        ss_res = K.sum(K.square(y_true-y_pred))
        ss_tot = K.sum(K.square(y_true-K.mean(y_true)))
        return (1 - ss_res/(ss_tot + K.epsilon()))
```
We define both metric and loss in a file called custom_model_elements.py. 
If the model is stored in the file K12.h5, as described above, then you can load the model as follows.
```python
from custom_model_elements import my_r_square_metric
    from custom_model_elements import my_mean_squared_error_weighted_genexp
    from tensorflow.keras.models import load_model
    
    model = load_model('K12.h5',
        custom_objects = {
            'my_r_square_metric': my_r_square_metric,
            'my_mean_squared_error_weighted_genexp': my_mean_squared_error_weighted_genexp,
            'loss': my_mean_squared_error_weighted_genexp()
        }
    )
```
---

### Input tensors of a loss function contain samples of an entire batch (not individual samples!)
To highlight the impact of y_true and y_pred containing batches of samples, rather than individual samples, let us look at the following custom loss function.
```python
   def loss_RMSE_by_batch(y_true, y_pred): 
      return tf.sqrt(tf.math.reduce_mean(tf.square(y_true - y_pred))) 
```

Constructing a sample-wise RMSE:
```python
   def loss_RMSE_by_sample(y_true, y_pred):
   
      # Reshape each tensor so that 1st dimension (batch dimension) stays intact, 
      # but all dimensions of an individual sample are flattened to 1D.
      y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
      y_pred = tf.reshape(y_pred, [tf.shape(y_true)[0], -1])
   
      # Now we apply mean only across the 2nd dimension, 
      # i.e. only across elements of each sample
      return K.sqrt( tf.math.reduce_mean(tf.square(y_true - y_pred), axis=[1]))
```

---
###  Mean applied automatically at the end of a loss function
The following definition 
```python
   def loss_MSE(y_true, y_pred): 
      return tf.math.reduce_mean(tf.square(y_true - y_pred)) 
```
is identical to
### 
```python
   def loss_MSE(y_true, y_pred): 
      return tf.square(y_true - y_pred)
```
---

### Feeding additional variables into the loss function
Example of function closure in Python:
```python
   # Nested functions.
   # Scope of outer function is inherited by inner function.

   # Outer function accepts whatever variables you want to give it (a.k.a. the context).
   # Here just one variable: x.

   def f(x):
  
      # Inner function implicitly INHERITS the context from the outer function.
      # Here: value of x.
      
      def g(y):
         # g is a function of y only, but we can nevertheless use x.
         return x * y
         
      return g

   # Create a function.
   g3 = f(3)
   
   # g3 now represents the function g with value x = 3.
   g3(5)   # What do you think the value is?
```

### Example of function closure to implement dual-weighted MSE as custom loss function:
```python
    def dual_weighted_mse(gamma_weight):
        def loss(target_tensor, prediction_tensor):
            return K.mean(
                (K.maximum(K.abs(target_tensor), K.abs(prediction_tensor)) ** gamma_weight) *
                (prediction_tensor - target_tensor) ** 2
            )

        return loss
    
    # How to use the loss function (with gamma = 5) when compiling a model:
    model.compile(loss=dual_weighted_mse(gamma_weight=5.), optimizer='adam')
```
---
### How to feed sample-specific information into a loss function

**Note:  This is an extended version of the code snippet in the Guide - extended to provide a working example.**
```python
# Import Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate, Dense
print("Using TensorFlow version {}".format(tf.__version__))

# Set User Options
nData = 1000
#myWeights = (0.8, 1.2)  
myWeights = (1.0, 1.0)   # Use (1.0, 1.0) for MSE equivalent

# Custom Loss Function
def loss_mse_weighted_input(weights=(1., 1.)):
  """Mean Square Error Weighted On Select Data."""

  def loss_custom(y_true2D, y_pred):
      y_true = y_true2D[:,0]
      y_suppl = y_true2D[:,1]
      weightsT = tf.where(y_suppl < 1, weights[0], weights[1])
      return K.mean(tf.multiply(
                    weightsT, 
                    tf.square(tf.subtract(y_pred, y_true))
      ))
  return loss_custom

# Custom MSE Metric
def mse_ytrue2D(y_true2D, y_pred):
    """Custom Mean Squared Error (Ignore Extra Input)"""
    return K.mean(tf.square(tf.subtract(y_pred, y_true2D[:,0])))

# Simple Functional Model
def build_model_functional(
        nFeatures=1, learnRate=0.1,
        loss="mean_squared_error", 
        metrics=[tf.keras.metrics.MeanSquaredError(name="mse")],
        verbose=True):
    """Build/Compile Functional Linear Model"""

    input_shape = (nFeatures, )
    inLayer = Input(shape=input_shape, name="inputs")
    outLayer2D = Dense(1)(inLayer)
    outLayer = tf.reshape(outLayer2D, [-1])
    model = Model(inputs=inLayer, outputs=outLayer, name="LinF_Model")
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learnRate),
        loss=loss, metrics=metrics)
    if verbose:
        model.summary()

    return model

# Generate Random Data
YData = np.random.random((nData))
YTest = np.random.random((nData))
XData = np.multiply(np.random.uniform(0.8, 1.2, size=nData), YData)
XTest = np.multiply(np.random.uniform(0.8, 1.2, size=nData), YTest)
InData = np.random.randint(2, size=nData)
InTest = np.random.randint(2, size=nData)
YData2D = np.concatenate((YData.reshape(-1,1), InData.reshape(-1,1)), axis=-1)
YTest2D = np.concatenate((YTest.reshape(-1,1), InTest.reshape(-1,1)), axis=-1)
print("Created Data.")

# Build and Train Models
lossCustom = loss_mse_weighted_input(myWeights)
metricsCustom = [mse_ytrue2D]
modelCustom = build_model_functional(loss=lossCustom, 
                                     metrics=metricsCustom)
historyCustom = modelCustom.fit(XData, YData2D,
                                batch_size=32,
                                epochs=5,
                                verbose=1)
evalCustom = modelCustom.evaluate(XTest, YTest2D)
YPred = modelCustom.predict(XTest)
print("Finished Custom Model, MSE: {:.4f}".\
      format(evalCustom[0]))

# Print Results
print("Model Test Loss Results:")
print("  Evaluation Loss:         {:.5f}".format(evalCustom[0]))
print("  Post-Process Loss:       {:.5f}".format(K.eval(
                                    lossCustom(YTest2D, YPred))))

print("\nModel Test MSE Results:")
print("  Evaluation Custom MSE:   {:.5f}".format(evalCustom[1]))
print("  Post-Process Custom MSE: {:.5f}".format(mse_ytrue2D(
                                    YTest2D, YPred)))
```


---
## Using NN layer functionality in loss functions and metrics 

Sample use of pooling layer:
```python
   pool_1 = tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(1, 1),
                padding='valid'
            )
            
   y_true_pooled = pool_1(y_true)
```

Sample use of convolutional layer:
```python
    weight_matrix = numpy.array([
        [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902],
        [0.01330621, 0.0596343, 0.09832033, 0.0596343, 0.01330621],
        [0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823],
        [0.01330621, 0.0596343, 0.09832033, 0.0596343, 0.01330621],
        [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902]
    ])
    
    # Expand dimensions from 5 x 5 to 5 x 5 x 3 x 3.
    weight_matrix = numpy.expand_dims(weight_matrix, axis=-1)
    weight_matrix = numpy.repeat(weight_matrix, axis=-1, repeats=3)
    weight_matrix = numpy.expand_dims(weight_matrix, axis=-1)
    weight_matrix = numpy.repeat(weight_matrix, axis=-1, repeats=3)
    
    # Conv layer also needs one bias per input channel.  Make these all zero.
    bias_vector = numpy.array([0, 0, 0], dtype=float)
    
    conv_layer_object = keras.layers.Conv2D(
        filters=3, kernel_size=(5, 5), strides=(1, 1),
        padding='same', data_format='channels_last',
        activation=None, use_bias=True, trainable=False,
        weights=[weight_matrix, bias_vector]
    )
```

Sample use of image functions:
```python
    def my_mse_with_sobel(weight=0.0):
        def loss(y_true,y_pred):
        
            # This function assumes that both y_true and y_pred have no channel dimension.
            # For example, if the images are 2-D, y_true and y_pred have dimensions of
            # batch_size x num_rows x num_columns.  tf.expand_dims adds the channel
            # dimensions before applying the Sobel operator.
            
            edges = tf.image.sobel_edges(tf.expand_dims(y_pred,-1))
            dy_pred = edges[...,0,0]
            dx_pred = edges[...,0,1]
            
            edges = tf.image.sobel_edges(tf.expand_dims(y_true,-1))
            dy_true = edges[...,0,0]
            dx_true = edges[...,0,1]
            
            return K.mean(
                tf.square(tf.subtract(y_pred,y_true)) +
                weight*tf.square(tf.subtract(dy_pred,dy_true)) +
                weight*tf.square(tf.subtract(dx_pred,dx_true))
            )
        return loss
```
---


### How to implement *if* statement functionality when conditioning on *model-independent variables*: using *tf.where*
Example loss function using where for weighting zero and non-zero values differently in the loss function
```python
    def my_mean_sqaured_error_wtzero(weight=(1.0,1.0)):
        def loss(y_true,y_pred):
            ones_array = tf.ones_like(y_true)
            weights_for_zero = tf.multiply(ones_array,weight[0])
            weights_for_nonzero = tf.multiply(ones_array,weight[1])
            
            weights = tf.where(tf.greater(y_true,0),weights_for_nonzero,weights_for_zero)
            
            return K.mean(tf.multiply(weights,tf.square(tf.subtract(y_pred,y_true))))
        return loss
```

### How to implement *if* statement functionality when conditioning on *model-dependent* variables: using soft discretization or raw confidence scores

Hard discretization:
```python
    y_pred_binary = tf.where(y_pred > cutoff, 1.0, 0.0)
```

Soft discretization:
```python
    c = 5  # constant to tune speed of transition from \texttt{0} to \texttt{1} 
    y_pred_binary_approx = tf.math.sigmoid(c * (y_pred - cutoff))
```
---
# Critical success index (CSI)
### 
```python
    def csi(use_as_loss_function, use_soft_discretization,
            hard_discretization_threshold=None):
            
        def loss(target_tensor, prediction_tensor):
            if hard_discretization_threshold is not None:
                prediction_tensor = tf.where(
                    prediction_tensor >= hard_discretization_threshold, 1., 0.
                )
            elif use_soft_discretization:
                prediction_tensor = K.sigmoid(prediction_tensor)
        
            num_true_positives = K.sum(target_tensor * prediction_tensor)
            num_false_positives = K.sum((1 - target_tensor) * prediction_tensor)
            num_false_negatives = K.sum(target_tensor * (1 - prediction_tensor))
            
            denominator = (
                num_true_positives + num_false_positives + num_false_negatives +
                K.epsilon()
            )

            csi_value = num_true_positives / denominator
            
            if use_as_loss_function:
                return 1. - csi_value
            else:
                return csi_value
        
        return loss
    
    # How to use the loss function (with no discretization) when compiling a model:
    loss_function = csi(
        use_as_loss_function=True, use_soft_discretization=False,
        hard_discretization_threshold=None
    )
    model.compile(loss=loss_function, optimizer='adam')
```

# Intersection over Union (IOU)
```python
    def iou(use_as_loss_function, use_soft_discretization, which_class,
            hard_discretization_threshold=None):

        def loss(target_tensor, prediction_tensor):
            if hard_discretization_threshold is not None:
                prediction_tensor = tf.where(
                    prediction_tensor >= hard_discretization_threshold, 1., 0.
                )
            elif use_soft_discretization:
                prediction_tensor = K.sigmoid(prediction_tensor)
        
            intersection_tensor = K.sum(
                target_tensor[..., which_class] * prediction_tensor[..., which_class],
                axis=(1, 2)
            )
            union_tensor = (
                K.sum(target_tensor, axis=(1, 2)) +
                K.sum(prediction_tensor, axis=(1, 2)) -
                intersection_tensor
            )

            iou_value = K.mean(
                intersection_tensor / (union_tensor + K.epsilon())
            )

            if use_as_loss_function:
                return 1. - iou_value
            else:
                return iou_value
        
        return loss
    
    # How to use the loss function (with no discretization, for class 1)
    # when compiling a model:
    loss_function = iou(
        use_as_loss_function=True, use_soft_discretization=False, which_class=1,
        hard_discretization_threshold=None
    )
    model.compile(loss=loss_function, optimizer='adam')
```

# Dice coefficient

```python
    def dice_coeff(use_as_loss_function, use_soft_discretization, which_class,
                   hard_discretization_threshold=None):
    
        def loss(target_tensor, prediction_tensor):
            if hard_discretization_threshold is not None:
                prediction_tensor = tf.where(
                    prediction_tensor >= hard_discretization_threshold, 1., 0.
                )
            elif use_soft_discretization:
                prediction_tensor = K.sigmoid(prediction_tensor)
        
            intersection_tensor = K.sum(
                target_tensor[..., which_class] * prediction_tensor[..., which_class],
                axis=(1, 2)
            )
            
            # Replacing prediction_tensor with target_tensor here would work, since
            # they both have the same size.
            num_pixels_tensor = K.sum(K.ones_like(prediction_tensor), axis=(1, 2, 3))
            dice_value = K.mean(intersection_tensor / num_pixels_tensor)

            if use_as_loss_function:
                return 1. - dice_value
            else:
                return dice_value
        
        return loss
```

# Tversky coefficient
```python
    def tversky_coeff(use_as_loss_function, use_soft_discretization, which_class,
                      false_positive_weight, false_negative_weight,
                      hard_discretization_threshold=None):
                      
        def loss(target_tensor, prediction_tensor):
            if hard_discretization_threshold is not None:
                prediction_tensor = tf.where(
                    prediction_tensor >= hard_discretization_threshold, 1., 0.
                )
            elif use_soft_discretization:
                prediction_tensor = K.sigmoid(prediction_tensor)
        
            intersection_tensor = K.sum(
                target_tensor[..., which_class] * prediction_tensor[..., which_class],
                axis=(1, 2)
            )
            false_positive_tensor = K.sum(
                (1 - target_tensor[..., which_class]) * prediction_tensor[..., which_class],
                axis=(1, 2)
            )
            false_negative_tensor = K.sum(
                target_tensor[..., which_class] * (1 - prediction_tensor[..., which_class]),
                axis=(1, 2)
            )
            denominator_tensor = (
                intersection_tensor + false_positive_tensor + false_negative_tensor +
                K.epsilon()
            )
            tversky_value = K.mean(intersection_tensor / denominator_tensor)

            if use_as_loss_function:
                return 1. - tversky_value
            else:
                return tversky_value
        
        return loss
```

# Fractions skill score (FSS)
```python
# Function to calculate "fractions skill score" (FSS).
#
# Function can be used as loss function or metric in neural networks.
#
# Implements FSS formula according to original FSS paper:
#    N.M. Roberts and H.W. Lean, "Scale-Selective Verification of
#    Rainfall Accumulation from High-Resolution Forecasts of Convective Events",
#    Monthly Weather Review, 2008.
# This paper is referred to as [RL08] in the code below.
    
def make_FSS_loss(mask_size):  # choose any mask size for calculating densities

    def my_FSS_loss(y_true, y_pred):

        # First: DISCRETIZE y_true and y_pred to have only binary values 0/1 
        # (or close to those for soft discretization)
        want_hard_discretization = False

        # This example assumes that y_true, y_pred have the shape (None, N, N, 1).
        
        cutoff = 0.5  # choose the cut off value for discretization

        if (want_hard_discretization):
           # Hard discretization:
           # can use that in metric, but not in loss
           y_true_binary = tf.where(y_true>cutoff, 1.0, 0.0)
           y_pred_binary = tf.where(y_pred>cutoff, 1.0, 0.0)

        else:
           # Soft discretization
           c = 10 # make sigmoid function steep
           y_true_binary = tf.math.sigmoid( c * ( y_true - cutoff ))
           y_pred_binary = tf.math.sigmoid( c * ( y_pred - cutoff ))

        # Done with discretization.

        # To calculate densities: apply average pooling to y_true.
        # Result is O(mask_size)(i,j) in Eq. (2) of [RL08].
        # Since we use AveragePooling, this automatically includes the factor 1/n^2 in Eq. (2).
        pool1 = tf.keras.layers.AveragePooling2D(pool_size=(mask_size, mask_size), strides=(1, 1), 
           padding='valid')
        y_true_density = pool1(y_true_binary);
        # Need to know for normalization later how many pixels there are after pooling
        n_density_pixels = tf.cast( (tf.shape(y_true_density)[1] * tf.shape(y_true_density)[2]) , 
           tf.float32 )

        # To calculate densities: apply average pooling to y_pred.
        # Result is M(mask_size)(i,j) in Eq. (3) of [RL08].
        # Since we use AveragePooling, this automatically includes the factor 1/n^2 in Eq. (3).
        pool2 = tf.keras.layers.AveragePooling2D(pool_size=(mask_size, mask_size),
                                                 strides=(1, 1), padding='valid')
        y_pred_density = pool2(y_pred_binary);

        # This calculates MSE(n) in Eq. (5) of [RL08].
        # Since we use MSE function, this automatically includes the factor 1/(Nx*Ny) in Eq. (5).
        MSE_n = tf.keras.losses.MeanSquaredError()(y_true_density, y_pred_density)

        # To calculate MSE_n_ref in Eq. (7) of [RL08] efficiently:
        # multiply each image with itself to get square terms, then sum up those terms.

        # Part 1 - calculate sum( O(n)i,j^2
        # Take y_true_densities as image and multiply image by itself.
        O_n_squared_image = tf.keras.layers.Multiply()([y_true_density, y_true_density])
        # Flatten result, to make it easier to sum over it.
        O_n_squared_vector = tf.keras.layers.Flatten()(O_n_squared_image)
        # Calculate sum over all terms.
        O_n_squared_sum = tf.reduce_sum(O_n_squared_vector)

        # Same for y_pred densitites:
        # Multiply image by itself
        M_n_squared_image = tf.keras.layers.Multiply()([y_pred_density, y_pred_density])
        # Flatten result, to make it easier to sum over it.
        M_n_squared_vector = tf.keras.layers.Flatten()(M_n_squared_image)
        # Calculate sum over all terms.
        M_n_squared_sum = tf.reduce_sum(M_n_squared_vector)
    
        MSE_n_ref = (O_n_squared_sum + M_n_squared_sum) / n_density_pixels
        
        # FSS score according to Eq. (6) of [RL08].
        # FSS = 1 - (MSE_n / MSE_n_ref)

        # FSS is a number between 0 and 1, with maximum of 1 (optimal value).
        # In loss functions: We want to MAXIMIZE FSS (best value is 1), 
        # so return only the last term to minimize.

        # Avoid division by zero if MSE_n_ref == 0
        # MSE_n_ref = 0 only if both input images contain only zeros.
        # In that case both images match exactly, i.e. we should return 0.
        my_epsilon = tf.keras.backend.epsilon()  # this is 10^(-7)

        if (want_hard_discretization):
           if MSE_n_ref == 0:
              return( MSE_n )
           else:
              return( MSE_n / MSE_n_ref )
        else:
           return (MSE_n / (MSE_n_ref + my_epsilon) )

    return my_FSS_loss
```
---
# Adaotive loss functions 
Discrete phases
```python
   # Standard MSE loss function
   def my_MSE_per_pixel( y_true, y_pred ):
      return K.square(y_pred - y_true)
   
   # Standard MSE loss function plus term penalizing only misses 
   def my_MSE_fewer_misses ( y_true, y_pred ):
      return K.square(y_pred - y_true) + K.maximum((y_true - y_pred), 0)
```

```python
   ### Training Phase 1
   # Assign first set of loss functions, optimizer and metrics:
   metric_list = [
       'mean_squared_error', my_count_true_convection, my_count_pred_convection, 
       my_overlap_count, jaccard_distance_loss
   ]
   model.compile(
       loss=my_MSE_per_pixel, metrics=metric_list, optimizer=RMSprop()
   )

   # First training phase (100 epochs from scratch):
   history = model.fit(
       [x_train_vis,x_train_ir], y_train, 
       validation_data=([x_test_vis,x_test_ir], y_test),
       epochs=100, batch_size=10
   )

   ### Training Phase 2
   # Assign second set of loss functions, optimizer and metrics:
   model.compile(
       loss=my_MSE_fewer_misses, metrics=metric_list, optimizer=RMSprop()
   )

   # Second training phase (fine tune model from Phase 1 with new loss function):
   history = model.fit(
       [x_train_vis,x_train_ir], y_train,
       validation_data=([x_test_vis,x_test_ir], y_test),
       epochs=100, batch_size=10
   )
```
---

# Structural Similarity Index Measure (SSIM)
### 
```python
   tf.image.ssim(
      img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03
   )
```

### 
```python
   tf.image.ssim_multiscale(
      img1, img2, max_val, power_factors=_MSSSIM_WEIGHTS, filter_size=11,
      filter_sigma=1.5, k1=0.01, k2=0.03
   )
```

### 
```python
```

### 
```python
```

### 
```python
```

### 
```python
```

### 
```python
```

### 
```python
```






```python

```

