# Complex Systems Simulation
## Species area curve
## Description

This project looks into the species area curve and consists of two main research questions. The first one is: what components are necessary to produce the power-law species are curve. The second is: how does the way organisms reproduce affect the shape of the species area curve. The algorithm that is used to simulate the dynamics, is the Voter model with mutation. The single grid simulations use the backwards Voter model, the island model uses the forward Voter model. 

## Usage
#### Structure
- **/Backwards voter model**: contains all runs with backwards voter model
- **/Basic model runs**: outcomes of every basic model run
- **/classes**: all the classes used for the multiple island model
- **/Forward voter model**: forward voter model, only used for island model
- **/Varying eta runs**: results for multiple varying eta runs
- **/Varying width runs**: results for varying the width
  
#### Install dependencies
```
pip3 install -r requirements.txt
```
#### Reproduction

To reproduce the island model results the simulation function in the multiple_islands.ipynb can be used.

## Authors and acknowledgment
- Jonas Argelo
- Jade Dubbeld
- Sjoerd Dronkers
