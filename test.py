from bayesianopt import SusiBO

TEST = 1  # Choose the model
susi = SusiBO(test=TEST, init_points=0, n_iter=4)
susi.run()


