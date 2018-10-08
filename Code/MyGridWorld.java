package assignment4;

import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.imageio.ImageIO;

import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.behavior.valuefunction.ValueFunctionInitialization;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.oomdp.auxiliary.common.ConstantStateGenerator;
import burlap.oomdp.auxiliary.common.SinglePFTF;
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
import burlap.oomdp.visualizer.Visualizer;

public class MyGridWorld {
    private GridWorldDomain gw;
    private Domain domain;
    private State initialState;
    private SimpleHashableStateFactory hashingFactory;
    private TerminalFunction tf;
    private RewardFunction rf;
    private double discount;
    private int[][] coordinates;

    public MyGridWorld (int[][] coordinates, double successRate, double goalReward, double defaultReward, double discount){
        this.gw = new GridWorldDomain(coordinates.length, coordinates.length);
        gw.setMap(coordinates);
        this.coordinates = coordinates;
        gw.setProbSucceedTransitionDynamics(successRate);
        this.domain = gw.generateDomain();
        this.initialState = GridWorldDomain.getOneAgentOneLocationState(domain);
        GridWorldDomain.setAgent(initialState, 0, 0);
        GridWorldDomain.setLocation(initialState, 0, coordinates.length - 1, coordinates[0].length - 1);
        this.hashingFactory = new SimpleHashableStateFactory();
        tf = new SinglePFTF(domain.getPropFunction(GridWorldDomain.PFATLOCATION));
        rf = new GoalBasedRF(new TFGoalCondition(tf), goalReward, defaultReward);
        this.discount = discount;
    }


    public GridWorldDomain getWorld() {
        return this.gw;
    }

    public Domain getDomain() {
        return this.domain;
    }

    public State getInitialState() {
        return this.initialState;
    }

    public SimpleHashableStateFactory getHashingFactory() {
        return this.hashingFactory;
    }

    public TerminalFunction getTerminalFunction() {
        return this.tf;
    }

    public RewardFunction getRewardFunction() {
        return this.rf;
    }

    public void valueIteration(String outputPath) throws IOException{

        Planner planner = new ValueIteration(domain, rf, tf, discount, hashingFactory, 0.001, 100000);
        long startTime = System.currentTimeMillis();
        System.out.println("Start");
        Policy p = planner.planFromState(initialState);
        long estimatedTime = System.currentTimeMillis() - startTime;
        System.out.println("End");
        System.out.println("Time Elapsed: " + estimatedTime + " ms");

        p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "_vi");
        simpleValueFunctionVis((ValueFunction)planner, p, outputPath + "_vi");
    }


    public void policyIteration(String outputPath) throws IOException{

        Planner planner = new PolicyIteration(domain, rf, tf, discount, hashingFactory, 0.001, 100000, 100000);

        long startTime = System.currentTimeMillis();
        System.out.println("Start");
        Policy p = planner.planFromState(initialState);
        long estimatedTime = System.currentTimeMillis() - startTime;
        System.out.println("End");
        System.out.println("Time Elapsed: " + estimatedTime + " ms");

        p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "_pi");

        simpleValueFunctionVis((ValueFunction)planner, p, outputPath + "_pi");
    }

    public void QLearning(String outputPath, final double gamma, final double qInit, final double learningRate, final double epsilon){
        //initial state generator
        final ConstantStateGenerator sg = new ConstantStateGenerator(initialState);
        LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

            @Override
            public String getAgentName() {
                return "Q-learning";
            }

            @Override
            public LearningAgent generateAgent() {
                return new MyQLearning(domain, gamma, hashingFactory, qInit, learningRate,epsilon);
            }
        };
        //define learning environment
        SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf, sg);
        //define experiment
        MyLearningAlgorithmExperimenter exp = new MyLearningAlgorithmExperimenter(env,
                5, 2000, qLearningFactory);
        exp.setUpPlottingConfiguration(500, 500, 2, 1000, TrialMode.MOSTRECENTANDAVERAGE,
                PerformanceMetric.STEPSPEREPISODE,
                PerformanceMetric.AVERAGEEPISODEREWARD);
        //start experiment
        long startTime = System.currentTimeMillis();
        System.out.println("Start");
        exp.startExperiment();
        long estimatedTime = System.currentTimeMillis() - startTime;
        System.out.println("End");
        System.out.println("Time Elapsed: " + estimatedTime + " ms");
    }


    public void simpleValueFunctionVis(ValueFunction valueFunction, Policy p, String outputPath) throws IOException{

        List<State> allStates = StateReachability.getReachableStates(initialState,
                (SADomain)domain, hashingFactory);
        MyValueFunctionVisualizerGUI gui = MyGridWorldDomain.getGridWorldValueFunctionVisualization(
                allStates, valueFunction, p);
        gui.initGUI();
        BufferedImage image = new BufferedImage(gui.getWidth(), gui.getHeight(), BufferedImage.TYPE_INT_ARGB);
        Graphics g = image.getGraphics();
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        gui.printAll(g);
        ImageIO.write(image, "png", new File(outputPath + ".png"));

    }


    public void initGUI(){
        Visualizer v = GridWorldVisualizer.getVisualizer(coordinates);
        VisualExplorer exp = new VisualExplorer(domain, v, initialState);
        exp.addKeyAction("w", GridWorldDomain.ACTIONNORTH);
        exp.addKeyAction("s", GridWorldDomain.ACTIONSOUTH);
        exp.addKeyAction("a", GridWorldDomain.ACTIONWEST);
        exp.addKeyAction("d", GridWorldDomain.ACTIONEAST);

        exp.initGUI();
    }
}