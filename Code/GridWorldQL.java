package assignment4;

import burlap.behavior.learningrate.SoftTimeInverseDecayLR;
import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.GreedyDeterministicQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.RandomPolicy;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.valuefunction.ValueFunctionInitialization;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.oomdp.auxiliary.common.ConstantStateGenerator;
import burlap.oomdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.singleagent.common.GoalBasedRF;
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;
import burlap.oomdp.statehashing.SimpleHashableStateFactory;
import org.apache.commons.lang3.CharUtils;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class GridWorldQL {
	public static void main(String [] args){
		Scanner reader = new Scanner(System.in);

		System.out.println("What domain is tested?\n1. 11 cells\n2. Four Rooms\n3. Maze");
		GridWorldDomain gwd = new GridWorldDomain(11, 11);
		gwd.setMapToFourRooms();
		int choice = reader.nextInt();

//        double[] discount = new double[] {0.9};
//        double[][] term_step_reward = new double[][] {{10, -0.1}};
//        double[] learning_rate = new double[] {0.1, 0.2, 0.4, 0.7, 1.0};
//        double[] qInit = new double[] {0.1, 1, 5, 10};
//        double[] epsilon = new double[] {0.01, 0.05, 0.1, 0.2};
		// below are for getting a figure of the best one
		double[] discount = new double[] {0.90};
		double[][] term_step_reward = new double[][] {{100, -0.1}};
		double[] learning_rate = new double[] {0.7};
		double[] qInit = new double[] {0.1};
		double[] epsilon = new double[] {0.01};


		//terminate in top right corner
		TerminalFunction tf = new GridWorldTerminalFunction(10, 10);
		int agentX = 0, agentY = 0;

		System.out.println(choice);
		switch (choice) {
			case 1:
				gwd = new GridWorldDomain(new int[][] {
						{ 1, 1, 0, 0, 0},
						{ 0, 1, 0, 0, 0},
						{ 0, 0, 0, 1, 0},
						{ 0, 1, 1, 1, 0},
						{ 0, 0, 0, 0, 0},
				});
				tf = new GridWorldTerminalFunction(4,4);
				agentX = 1;
				agentY = 0;
				break;
			case 2:
				gwd = new GridWorldDomain(11, 11);
				gwd.setMapToFourRooms();
				tf = new GridWorldTerminalFunction(10, 10);
				break;
			case 3:  // runs out of memory for Q Learning due to the large size of the maze
				gwd = new GridWorldDomain(new int[][] {
						{0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1},
						{0,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1},
						{0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1},
						{1,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1},
						{0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1},
						{0,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1},
						{0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,1},
						{1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,1,1},
						{0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,1},
						{0,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1},
						{0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1},
						{0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1},
						{0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1},
						{0,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1},
						{0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,1},
						{1,1,1,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1},
						{0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,1},
						{0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1},
						{0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,1},
						{0,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,1,1,0,1,1,1,0,1,0,1},
						{0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,1},
						{1,1,0,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1},
						{0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1},
						{0,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1},
						{0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,1},
						{0,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,1,1},
						{0,0,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,1},
						{1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1,0,1},
						{0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,0,0,1},
						{0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,0,1,1,1},
						{0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,1},
						{1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,1,1,1,1},
						{0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1},
						{0,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,0,1},
						{0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,1},
						{0,1,1,1,1,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1},
						{0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,1},
						{1,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,0,1},
						{0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1},
						{0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
				});
				tf = new GridWorldTerminalFunction(0, 29);
				agentX = 29;
				agentY = 0;
				break;
		}

//        System.out.println("What's the initial learning rate?");
//
//        double initLR= reader.nextDouble();

		// System.out.println("What's the soft factor?");

		// int sFactor = reader.nextInt();

		//only go in intended directon 80% of the time
		gwd.setProbSucceedTransitionDynamics(0.8);

		final Domain domain = gwd.generateDomain();

		//get initial state with agent in 0,0
		final State s = GridWorldDomain.getOneAgentNoLocationState(domain);
		GridWorldDomain.setAgent(s, agentX, agentY);

//        System.out.println("What's the number of episodes?");
//        final int numIter = reader.nextInt();
		final int numIter = 2000;
//
//        RewardFunction rf = new GoalBasedRF(new TFGoalCondition(tf), term_step_reward[0][0], term_step_reward[0][1]);
//        //create environment
//        //SimulatedEnvironment env = new SimulatedEnvironment(domain,rf, tf, s);
//        final ConstantStateGenerator sg = new ConstantStateGenerator(s);
//        SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();
//
//        String head = "Initial Q";
//
//        new File("./output/"+head).mkdirs();
//        File file = new File("./output/"+head+"/out_BD2"); //Your file
//        try {
//            FileOutputStream fos = new FileOutputStream(file);
//            PrintStream ps = new PrintStream(fos);
//            System.setOut(ps);
//        } catch (Exception yuck) {
//            System.out.println("ehh");
//        }
//        OutputStream outs = System.out;
//        PrintStream dos = new PrintStream(outs);
//
//        SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf, sg);
//        head = "Initial Q";
//        System.setOut(dos);
//        System.out.println(head);
//        new File("./output/"+head).mkdirs();
//        file = new File("./output/"+head+"/out_BD"); //Your file
//
//
//        LearningAgentFactory qLearningFactory10 = new LearningAgentFactory() {
//            @Override
//            public String getAgentName() {
//                return "Q-learning with Q-Init " + qInit[0];
//            }
//            @Override
//            public LearningAgent generateAgent() {
//                return new MyQLearning(domain, discount[0], hashingFactory, qInit[0], learning_rate[0], epsilon[0]);
//                // Domain domain, double gamma, HashableStateFactory hashingFactory,
//                // double qInit, double learningRate, double epsilon
//            }
//        };
//        LearningAgentFactory qLearningFactory11 = new LearningAgentFactory() {
//            @Override
//            public String getAgentName() {
//                return "Q-learning with Q-Init " +qInit[1];
//            }
//            @Override
//            public LearningAgent generateAgent() {
//                return new MyQLearning(domain, discount[0], hashingFactory, qInit[1], learning_rate[0], epsilon[0]);
//            }
//        };
//        LearningAgentFactory qLearningFactory12 = new LearningAgentFactory() {
//            @Override
//            public String getAgentName() {
//                return "Q-learning with Q-Init " +qInit[2];
//            }
//            @Override
//            public LearningAgent generateAgent() {
//                return new MyQLearning(domain, discount[0], hashingFactory, qInit[2], learning_rate[0], epsilon[0]);
//            }
//        };
//        LearningAgentFactory qLearningFactory13 = new LearningAgentFactory() {
//            @Override
//            public String getAgentName() {
//                return "Q-learning with Q-Init " +qInit[3];
//            }
//            @Override
//            public LearningAgent generateAgent() {
//                return new MyQLearning(domain, discount[0], hashingFactory, qInit[3], learning_rate[0], epsilon[0]);
//            }
//        };
//        MyLearningAlgorithmExperimenter exp3 = new MyLearningAlgorithmExperimenter(env,
//                5, 2000, qLearningFactory10,qLearningFactory11,qLearningFactory12,qLearningFactory13);
//        exp3.setUpPlottingConfiguration(500, 500, 2, 1000, TrialMode.MOSTRECENTANDAVERAGE,
//                PerformanceMetric.STEPSPEREPISODE,
//                PerformanceMetric.AVERAGEEPISODEREWARD);
//        //start experiment
//        long startTime = System.currentTimeMillis();
//        System.out.println("Start");
//        exp3.startExperiment();
//        long estimatedTime = System.currentTimeMillis() - startTime;
//        System.out.println("End");
//        System.out.println("Time Elapsed: " + estimatedTime + " ms");
//
//


//
//        System.out.println("What's the initial Q value?");
//        final double qInit = reader.nextDouble();
//
//        //System.out.println("What's the discount factor?");
//       //final double gamma = reader.nextDouble();
//        ArrayList<double[]> results = new ArrayList<>();
//
//        XYSeriesCollection dataset = new XYSeriesCollection();

		System.out.println("Terminating Reward, Step Reward, Discount Factor, Learning Rate, Q_Init, Epsilon, Steps to Exit, Runtime (in ms), Total Reward");
		for (double[] r : term_step_reward) {
			for (double d : discount) {
				for (double lr : learning_rate) {
					for (double q : qInit) {
						for (double e: epsilon) {
							//all transitions return -1
							RewardFunction rf = new GoalBasedRF(new TFGoalCondition(tf), r[0], r[1]);
							//create environment
							SimulatedEnvironment env = new SimulatedEnvironment(domain,rf, tf, s);
							QLearning ql = new QLearning(domain, d, new SimpleHashableStateFactory(), q, lr);
							//ql.setLearningRateFunction(new SoftTimeInverseDecayLR(initLR, sFactor));
							ql.setLearningPolicy(new EpsilonGreedy(ql, e));
							ql.setQInitFunction(new ValueFunctionInitialization.ConstantValueFunctionInitialization());
							final long startTime = System.currentTimeMillis();
							ql.initializeForPlanning(rf, tf, numIter);

							XYSeries steps = new XYSeries("Q-Learning with epsilon " + e);

							for(int i = 0; i < numIter; i++){
								EpisodeAnalysis ea = ql.runLearningEpisode(env, 100000);
								steps.add(i, ea.actionSequence.size());
								env.resetEnvironment();
							}

							//dataset.addSeries(steps);

							Policy p = ql.planFromState(s);
							final long endTime = System.currentTimeMillis();
							EpisodeAnalysis ea = p.evaluateBehavior(s, rf, tf);
							double sum = 0;
							for(Double rs : ea.rewardSequence) {
								sum += rs;
							}
							double[] res = new double[]{r[0], r[1], d, lr, q, e, ea.actionSequence.size(), (endTime-startTime), sum};
							// results.add(res);
							System.out.println(Arrays.toString(res).substring(1, Arrays.toString(res).length() - 2));
							List<State> allStates = StateReachability.getReachableStates(s, (SADomain) domain,
									new SimpleHashableStateFactory());

							ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
									allStates, ql, p);
							gui.initGUI();
						}
					}
				}
			}
		}
//
//        JFreeChart xyLineChart = ChartFactory.createXYLineChart(
//                "Q-Learning Stepts to Exit vs Episodes",
//                "Configurations",
//                "Steps",
//                dataset,
//                PlotOrientation.VERTICAL,
//                true, true, false);
//
//        int width = 640;
//        int height = 480;
//        File XYChart = new File("target/SGW_" + "epsilon.jpeg");
//        try {
//            ChartUtilities.saveChartAsJPEG(XYChart, xyLineChart, width, height);
//        } catch (IOException io) {
//            System.out.println("couldn't save file");
//        }


//        for (double[] result : results) {
//            StringBuilder sb = new StringBuilder();
//            for (double val : result) {
//                sb.append(val + ",");
//            }
//            System.out.println(sb.subSequence(0, sb.length()-1));
//        }


		// System.out.println("Steps taken to exit: " + ea.actionSequence.size());
		// System.out.println("Total Reward: " + sum);

		//System.out.println("Steps taken to exit: " + ea.actionSequence.size());


	}

}
