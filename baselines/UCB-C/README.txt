This code implements the experiments in the project: Optimistic Bayesian Optimization with Unknown Constraints.

-----
To conduct the experiment, execute exec_bo.py script with the corresponding acquisition function and the experiment:

  python exec_bo.py expm/"$EXPR".pkl --initseed "$SEED" --random "$RAND" --acqfunc "$ACFC"

where $EXPR is the experiment name: "S_A0", "S_A1", "S_A2", "gas", "cnn", "qchip"
      $SEED is the random seed, set to 0
	  $RAND is the the number random experiments
	  $ACFC is the acquisition function: "ucbd", "ucbc", "eic", "cmes_ibo".

-----
To conduct the experiment with ADMMBO, execute the following command:

  python exec_admmbo.py expm/"$EXPR".pkl --initseed "$SEED" --random "$RAND"
