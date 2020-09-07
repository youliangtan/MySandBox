# Simple VRP 
scaffolding code to test out VRP, task allocation and optimization problem via heuristics and constraints.

```bash
# Run Cpp Task allo script
g++ -o toy_problem toy_problem.cpp && ./toy_problem

# Run python script
pip3 install ortools
python3 delivery_cp.py
```

To Validate Task allocation Data via viz:
 - Edit `task_config.yaml` and `allocation.yaml`
 - Run this:
    ```bash
    python3 task_allocation_viz.py
    ```