# Complex Systems Simulation
## Species area curve
## Description

This project looks into the species area curve and consists of two main research questions.
1. What components are necessary to produce the power-law species are curve? 
2. How does the way organisms reproduce affect the shape of the species area curve? 

The algorithm that is used to simulate the dynamics, is the Voter model with mutation. The single grid simulations use the backwards Voter model, the island model uses the forward Voter model. In the single grid, varying values for the fatness of the fat tail distribution and varying kernel widths are tested. In the multiple island model two islands are places increaslingly further apart from each other to test the change in dynamics.

## Usage
#### Structure
- **/Backwards voter model**: contains all runs with backwards voter model
- **/Basic model runs**: outcomes of every basic model run
- **/classes**: all the classes used for the multiple island model
- **/Forward voter model**: forward voter model, only used for island model
- **/Varying eta runs**: results for varying eta (backward voter model)
- **/Varying width runs**: results for varying the kernel width (backward voter model)

#### Install dependencies
```
pip3 install -r requirements.txt
```
#### Reproduction

To reproduce the island model results the simulation function in the multiple_islands.ipynb can be used.

To reproduce the results obtained for single grids, the function voter_model_fast() from the notebook 'Backwards voter model/backwards_time_voter_model.ipynb' can be used. This function is based on the paper 'A coalescence approach to spatial neutral ecology' by Rosindell J. et al (2008).

## Authors and acknowledgment
- Jonas Argelo
- Jade Dubbeld
- Sjoerd Dronkers
