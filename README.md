# SindyAutoencoders

Code for the paper ["Data-driven discovery of coordinates and governing equations"](https://arxiv.org/abs/1904.02107) by Kathleen Champion, Bethany Lusch, J. Nathan Kutz, and Steven L. Brunton.

The paper contains results for three example problems based on the Lorenz system, a reaction-diffusion system, and the nonlinear pendulum. Code for each example can be found in the respective subfolder in the examples folder. For each example, there are jupyter notebooks for (1) running the training procedure and (2) analyzing the resulting models.

Our training procedure can be replicated using the jupyter notebooks in each folder, which will run ten instances of the training and produce ten models. Running the notebook will also produce summary output, saved in a Pandas dataframe. The information saved in the dataframe can be used to compare among models. In the paper, we perform a model selection among the resulting models and select a subset of the models to highlight. Our model selection procedure is described in the appendix, along with a detailed description of the training procedure.

For each example, we also include jupyter notebooks to analyze the resulting models. These notebooks produce plots of the results and print out summary statistics on test data. The models analyzed in the paper are included in the repository.

Creating the network architecture and running the training procedure requires the specification of several parameters. A description of the parameters is as follows:

* `input_dim` - dimension of each sample of the input data
* `latent_dim` - dimension of the latent space
* `model_order` - either 1 or 2; determines whether the SINDy model predicts first or second order derivatives
* `poly_order` - maximum polynomial order to which to build the SINDy library; integer from 1-5
* `include_sine` - boolean, whether or not to include sine functions in the SINDy library
* `library_dim` - total number of library functions; this is determined based on the `latent_dim`, `model_order`, `poly_order`, and `include_sine` parameters and can be calculated using the function `library_side` in `sindy_utils.py`

* `sequential_thresholding` - boolean, whether or not to perform sequential thresholding on the SINDy coefficient matrix
* `coefficient_threshold` - float, minimum magnitude of coefficients to keep in the SINDy coefficient matrix when performing thresholding
*  `threshold_frequency` - integer, number of epochs after which to perform thresholding
* `coefficient_mask` - matrix of ones and zeros that determines which coefficients are still included in the SINDy model; typically initialized to all ones and will be modified by the sequential thresholding procedure
* `coefficient_initialization` - how to initialize the SINDy coefficient matrix; options are `'constant'` (initialize as all 1s), `'xavier'` (initialize using the xavier initialization approach), `'specified'` (pass in an additional parameter `init_coefficients` that has the values to use to initialize the SINDy coefficient matrix)

* `loss_weight_decoder` - float, weighting of the autoencoder reconstruction in the loss function (should keep this at 1.0 and adjust the other weightings proportionally)
* `loss_weight_sindy_z`- float, weighting of the SINDy prediction in the latent space in the loss function
* `loss_weight_sindy_x` - float, weighting of the SINDy prediction passed back to the input space in the loss function
* `loss_weight_sindy_regularization` - float, weighting of the L1 regularization on the SINDy coefficients in the loss function

* `activation` - activation function to be used in the network; options are `'sigmoid'`, `'relu'`, `'linear'`, or `'elu'`
* `widths` - list of ints specifying the number of units for each layer of the encoder; decoder widths will be the reverse order of these widths

* `epoch_size` - number of training samples in an epoch
* `batch_size` - number of samples to use in a batch of training
* `learning rate` - float; learning rate passed to the adam optimizer
* `data_path` - path specifying where to save the resulting models
* `print_progress` - boolean, whether or not to print updates during training
* `print_frequency` - print progress at intervals of this many epochs
* `max_epochs` - how many epochs to run the training procedure for
* `refinement_epochs` - how many epochs to run the refinement training for (see paper for description of the refinement procedure)

## Resources

- Carini, M., Auteri, F., & Giannetti, F. (2015). [Centre-manifold reduction of bifurcating flows](https://doi.org/10.1017/jfm.2015.3). *J. Fluid Mech.*, 767, 109-145. Cambridge University Press. DOI: [10.1017/jfm.2015.3](https://doi.org/10.1017/jfm.2015.3).

- Champion, K., Lusch, B., Kutz, J. N., & Brunton, S. L. (2019). [Data-driven discovery of coordinates and governing equations](https://doi.org/10.1073/pnas.1906995116). *Proceedings of the National Academy of Sciences of the United States of America*, 116(45), 22445-22451. National Academy of Sciences. DOI: [10.1073/pnas.1906995116](https://doi.org/10.1073/pnas.1906995116).

- Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). [Discovering governing equations from data by sparse identification of nonlinear dynamical systems](https://doi.org/10.1073/pnas.1517384113). *Proceedings of the National Academy of Sciences of the United States of America*, 113(15), 3932-3937. National Academy of Sciences. DOI: [10.1073/pnas.1517384113](https://doi.org/10.1073/pnas.1517384113).

- Robinson, H., Pawar, S., Rasheed, A., & San, O. (2022). [Physics guided neural networks for modelling of non-linear dynamics](https://doi.org/10.1016/j.neunet.2022.07.023). *Neural Networks*, 154, 333-345. Elsevier Ltd. DOI: [10.1016/j.neunet.2022.07.023](https://doi.org/10.1016/j.neunet.2022.07.023).

- Champion, K., Lusch, B., Kutz, J. N., & Brunton, S. L. (2019). [SindyAutoencoders](https://github.com/kpchamp/SindyAutoencoders). GitHub.

- Conti, P., Gobat, G., Fresca, S., Manzoni, A., & Frangi, A. (2023). [Reduced order modeling of parametrized systems through autoencoders and SINDy approach: continuation of periodic solutions](https://doi.org/10.1016/j.cma.2023.116072). *Computer Methods in Applied Mechanics and Engineering*, 411, 116072. Elsevier Ltd. DOI: [10.1016/j.cma.2023.116072](https://doi.org/10.1016/j.cma.2023.116072).

- Champion, K. P., Brunton, S. L., & Kutz, J. N. (2019). [Discovery of Nonlinear Multiscale Systems: Sampling Strategies and Embeddings](https://doi.org/10.1137/18M1188227). *SIAM Journal on Applied Dynamical Systems*, 18(1), 312-333. SIAM. DOI: [10.1137/18M1188227](https://doi.org/10.1137/18M1188227).

- Otto, S. E., & Rowley, C. W. (2019). [Linearly Recurrent Autoencoder Networks for Learning Dynamics](https://doi.org/10.1137/18M1177846). *SIAM Journal on Applied Dynamical Systems*, 18(1), 558-593. SIAM. DOI: [10.1137/18M1177846](https://doi.org/10.1137/18M1177846).

