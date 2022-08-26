# Probabilistic Spiking Neural Network Research Internship

## References

### Papers

- [Probabilistic Spiking Neural Networks](https://arxiv.org/pdf/1910.01059.pdf)
- [Expectation Maximization for PSSNs](https://arxiv.org/pdf/2102.03280.pdf)
- [Variational Bayes](https://arxiv.org/pdf/2103.01327.pdf) *Algorthim 4*
- [Variational Auto-Encoders](https://arxiv.org/pdf/1906.02691.pdf) *Useful for understanding Partially Observed training methods*

### Resources

- [Norse Pytorch SSN Library](https://github.com/norse/norse)

## Abstract

This repo contains code and results I obtained as part of an internship researching SNNs

- **PSSN/** Julia PSSN implementation
- **Norse-Pytorch/** Norse demo for comparison
- **TestData.xlsx** Training results and parameters. Useful for comparing alternative implementations
- **PSSN_Presentation.pptx** A presentations with some visuals describing obtained results and findings

## Set Up

- Run the PSSN/DataEncoding.jl file to generate the encoded data
- Run Norse-Pytorch/norse-3layer.ipynb to train the norse demo, you can select Iris or MNIST dataset
- Run PSSN/Main.jl to train the julia model. PSSN/PSSN.jl contains the various funtions and structs to define and train a network

## Next Steps

- Test PSSN implementation on more diverse training problems
