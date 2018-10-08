package assignment4;

import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
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
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;
import burlap.oomdp.statehashing.SimpleHashableStateFactory;
import burlap.oomdp.visualizer.Visualizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class GridWorldPI{
    public static void main(String [] args){
        Scanner reader = new Scanner(System.in);

        System.out.println("What domain is tested?\n1. 11 cells\n2. Four Rooms\n3. Maze");
        GridWorldDomain gwd = new GridWorldDomain(11, 11);
        gwd.setMapToFourRooms();

        //double[][] term_step_reward = new double[][] {{5000, -0.01}, {1000, -0.1}, {1000, -0.01}, {2000, -0.01}, {2000, -0.1}, {10000, -0.01}};
        //double[][] term_step_reward = new double[][] {{5000, -0.1}, {10000, -0.1}};
        double[][] term_step_reward = new double[][] {{10, -0.1}};
        double[] discount = new double[] {0.99, 0.95, 0.9};

        //terminate in top right corner
        TerminalFunction tf = new GridWorldTerminalFunction(10, 10);
        int agentX = 0, agentY = 0;

        switch (reader.nextInt()) {
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
            case 3:
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


        ArrayList<double[]> results = new ArrayList<>();

        for (double[] r : term_step_reward) {
            for (double d : discount) {
                System.out.println(r[0]);
                System.out.println(r[1]);

                gwd.setProbSucceedTransitionDynamics(0.8);

                Domain domain = gwd.generateDomain();
                //get initial state with agent in 0,0
                State s = GridWorldDomain.getOneAgentNoLocationState(domain);
                GridWorldDomain.setAgent(s, agentX, agentY);

                RewardFunction rf = new GoalBasedRF(new TFGoalCondition(tf), r[0], r[1]);
                PolicyIteration pi = new PolicyIteration(domain, rf, tf, d, new SimpleHashableStateFactory(),
                        0.01, 1000, 100);

                final long startTime = System.currentTimeMillis();


                //run planning from our initial state
                Policy p = pi.planFromState(s);
                final long endTime = System.currentTimeMillis();
                EpisodeAnalysis ea = p.evaluateBehavior(s, rf, tf, 100000);

                double sum = 0;
                for(Double rs : ea.rewardSequence) {
                    sum += rs;
                }
//                Visualizer v = GridWorldVisualizer.getVisualizer(gwd.getMap());
//                new EpisodeSequenceVisualizer(v, domain, Arrays.asList(p.evaluateBehavior(s, rf, tf)));
//
//
//                List<State> allStates = StateReachability.getReachableStates(s,
//                        (SADomain) domain, new SimpleHashableStateFactory());
//
//                ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
//                        allStates, pi, p);
//                 gui.initGUI();

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


    }

}
