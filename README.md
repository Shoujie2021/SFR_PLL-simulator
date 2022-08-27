# SFR_PLL-simulator
The python simulation for Synchronous Reference Frame Phase Locked Loop

# Environment setup
  conda create --name power
  
  conda activate power
  
  conda config --add channels conda-forge
  
  conda install click matplotlib
  
# Introduction
The program generates a 3 phase voltage signal, executes a DQ0 transform, then
tracks the voltage angle with a PI controller

# Run
python main.py run -n 3

