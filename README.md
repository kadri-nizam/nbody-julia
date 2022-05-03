# nbody-julia
N-body simulation written in Julia for ASTRO577 for the purpose of studying TTV.

# Installation
Clone the repository and navigate to the folder in a terminal. In the terminal:
``` 
  julia --project=. 
```
This will start the Julia REPL. The project then needs to be instantiated. Press the `]` key in the REPL to change to Pkg mode.
Download all the necessary packages for the code to run by entering `instantiate`. Once it is done, backspace out of the Pkg environment back into REPL.

# Opening the Pluto notebook
In the REPL environment, type:
``` 
  using Pluto; Pluto.run() 
```
This will start a notebook session on your browser. In the textbox, type:
```
  ./nbody.jl
```
Assuming that you are in the cloned directory.
