# Notes 
unitization depends on a few libraries:  
* tensorflow == 1.6.0
* keras == 2.1.5
* jupyter
* numpy  

unitization is tested in the environment with tensorflow == 1.6.0 and keras == 2.1.5. The higher versions of these libraries might work.

# Usage
1. Install the required dependencies like tensorflow and keras
2. Run Jupyter Notebook
3. Open `main.ipynb` on Jupyter
4. Run all the cells, and the results will be shown in the last cell
5. You can use a different `seed` in the second cell to test the code
6. You can delete `weights` folder that includes the initial weights, and run the code with a different `seed`
7. Delete `test_accuracy.json` that stores all the results if you run the code again
