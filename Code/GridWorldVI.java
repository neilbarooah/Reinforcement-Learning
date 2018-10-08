package assignment4;

import burlap.behavior.policy.Policy;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.oomdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.singleagent.common.GoalBasedRF;
import burlap.oomdp.singleagent.common.UniformCostRF;
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;
import burlap.oomdp.singleagent.explorer.VisualExplorer;
import burlap.oomdp.statehashing.SimpleHashableStateFactory;
import burlap.oomdp.visualizer.StateRenderLayer;
import burlap.oomdp.visualizer.Visualizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

import assignment4.util.AgentPainter;
import assignment4.util.LocationPainter;
import assignment4.util.WallPainter;

public class GridWorldVI{
    public static void main(String [] args) throws IOException{
        Scanner reader = new Scanner(System.in);
        String CLASSAGENT = "agent";
    	String CLASSLOCATION = "location";

        System.out.println("What domain is tested?\n1. 11 cells\n2. Four Rooms\n3. Maze");
        GridWorldDomain gwd = new GridWorldDomain(11, 11);
        gwd.setMapToFourRooms();

        double[][] term_step_reward = new double[][] {{100, -0.1}};
        double[] discount = new double[] {0.99, 0.95, 0.9};

        //terminate in top right corner
        TerminalFunction tf = new GridWorldTerminalFunction(10, 10);
        int agentX = 0, agentY = 0;
        int[][] map = null;
        String header = null;

        switch (reader.nextInt()) {
            case 1:
            	header = "singleBlock";
            	map = new int[][] {
                        { 1, 1, 0, 0, 0},
                        { 0, 1, 0, 0, 0},
                        { 0, 0, 0, 1, 0},
                        { 0, 1, 1, 1, 0},
                        { 0, 0, 0, 0, 0},
            	};
                gwd = new GridWorldDomain(map);
                tf = new GridWorldTerminalFunction(map[0].length-1,map[1].length-1);
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

//        System.out.println("What's the termination reward?");
//        final double tr = reader.nextDouble();
//
//        System.out.println("What's the step reward?");
//        final double sr = reader.nextDouble();
//
//        System.out.println("What is the discount?");
//        final double dc = reader.nextDouble();


        //only go in intended directon 80% of the time
        gwd.setProbSucceedTransitionDynamics(0.8);

        Domain domain = gwd.generateDomain();

        //get initial state with agent in 0,0
        State s = GridWorldDomain.getOneAgentNoLocationState(domain);
        GridWorldDomain.setAgent(s, agentX, agentY);
        ArrayList<double[]> results = new ArrayList<>();

        //all transitions return -1
        for (double[] r : term_step_reward) {
            for (double d : discount) {
                RewardFunction rf = new GoalBasedRF(new TFGoalCondition(tf), r[0], r[1]);
                //initial view of GridWorld Domain
                SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf,
                        s);
                final long startTime = System.currentTimeMillis();
                //setup vi with 0.99 discount factor, a value
                //function initialization that initializes all states to value 0, and which will
                //run for 30 iterations over the state space
                ValueIteration vi = new ValueIteration(domain, rf, tf, d, new SimpleHashableStateFactory(),
                        0.01, 50000);
                //run planning from our initial state
                Policy p = vi.planFromState(s);
                final long endTime = System.currentTimeMillis();
                EpisodeAnalysis ea = p.evaluateBehavior(s, rf, tf, 100000);

                double sum = 0;
                for(Double rs : ea.rewardSequence) {
                    sum += rs;
                }
                double[] result = new double[] {r[0], r[1], d, ea.actionSequence.size(), (endTime - startTime), sum};
                results.add(result);
//                System.out.println("Terminating Reward: " + r[0] + ", Step Reward: " + r[1] + ", Discount Factor: " + d);
//                System.out.println("Steps taken to exit: " + ea.actionSequence.size());
//                System.out.println("Runtime: " + (endTime - startTime) + " milliseconds");
//                System.out.println("Total Reward: " + sum);


//                Visualizer v = GridWorldVisualizer.getVisualizer(gwd.getMap());
//                new EpisodeSequenceVisualizer(v, domain, Arrays.asList(ea));
//                List<State> allStates = StateReachability.getReachableStates(s,
//                        (SADomain) domain, new SimpleHashableStateFactory());
//
//                ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
//                        allStates, vi, p);
//                gui.initGUI();
            }
        }
        System.out.println("Terminating Reward, Step Reward, Discount Factor, Steps to Exit, Runtime (in ms), Total Reward");
        for (double[] result : results) {
            StringBuilder sb = new StringBuilder();
            for (double val : result) {
                sb.append(val + ",");
            }
            System.out.println(sb.subSequence(0, sb.length()-1));
        }

        

//        Visualizer v = new Visualizer(getStateRenderLayer(map,CLASSLOCATION,CLASSAGENT));
//        VisualExplorer exp = new VisualExplorer(domain, env, v);

//		exp.setTitle("Easy Grid World");
//		exp.initGUI();



//        
        

    }
    
    public static StateRenderLayer getStateRenderLayer(int[][] map, String CLASSLOCATION, String CLASSAGENT ) {
		StateRenderLayer rl = new StateRenderLayer();
		rl.addStaticPainter(new WallPainter(map));
		rl.addObjectClassPainter(CLASSLOCATION, new LocationPainter(map));
		rl.addObjectClassPainter(CLASSAGENT, new AgentPainter(map));
		return rl;
	}

}
