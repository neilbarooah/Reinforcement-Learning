CS 4641 - Machine Learning
Assignment 4: Markov Decision Processes

The structure of the directory is as follows:

1. nbarooah3-analysis.pdf - My Report
2. code/
	   - GridWorldPI.java - Java code to conduct Policy Iteration on all 3 MDPs.
	   - GridWorldVI.java - Java code to conduct Value Iteration on all 3 MDPs.
	   - GridWOrldQL.java - Java code to conduct Q-Learning on all 3 MDPs.
	   - All other files are supporting files for the grid, maps, plotting tools etc.
3. Figures/
		- Four Rooms/
					- Contains all figures pertaining to Four Rooms by parameters varied.
		- Maze/
					- Contains all figures pertaining to Maze by parameters varied.
		- SGW/
					- Contains all figures pertaining to Simple Grid World by parameters varied.
		- Policy Maps/
					- Contains all Policy Maps obtained from PI, VI, QL on all problems.


Please note that I have used Burlap to implement all the algorithms. I have not written any code in this assignment apart from changing parameters and adding my own variation of the grid (SGW).

In order to run the code, navigate to the code/ folder and run either of GridWorldPI.java, GridWorldVI.java or GridWorldQL.java based on the experiment you'd like to recreate. You must have Burlap and all dependencies installed in this directory in order to run these files. Also note that there is a lot of code that is commented out in these files based on parameters I have varied. You must uncomment the desired section based on what part of VI/QL/PI you'd like to run.