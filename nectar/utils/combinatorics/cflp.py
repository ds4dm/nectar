import copy

import numpy as np
from gurobipy import *

from nectar.utils import SIPP


class CFLP(SIPP):
    """
    Class for Capacitated Facility Location Problem
    A child class of Stochastic Integer Programming Problem
    n = Number of nodes
    Nscen = Number of scenarios
    c_f = Fixed cost of opening facility
    c_v = Variable cost of opening facility
    c_tf = Fixed transport cost
    c_tv = Variable transport cost
    d = Stochastic demand at nodes
    prob = Probability of each scenario
    """

    def __init__(self, data):
        self.num_nodes = 0
        self.c_f, self.c_v = np.asarray([]), np.asarray([])
        self.c_tf, self.c_tv = np.asarray([]), np.asarray([])
        self.Xi = np.asarray([])
        self.num_scenarios, self.MaxCap = 0, 0
        self.prob = np.asarray([])
        self.xi_bar = np.asarray([])

        self._initialize_instance(data)

    def _initialize_instance(self, data):
        self.num_nodes = data["c_f"].shape[0]
        self.c_f, self.c_v = data["c_f"], data["c_v"]
        self.c_tf, self.c_tv = data["c_tf"], data["c_tv"]
        self.Xi = data["scenario"]
        self.num_scenarios = self.Xi.shape[0]
        self.prob = data["prob"] if "prob" in data else np.array([1/self.num_scenarios] * self.num_scenarios)
        self.MaxCap = np.max(np.sum(self.Xi, axis=1))

    def set_xi_bar(self, xi_bar):
        self.xi_bar = np.asarray([xi_bar])

    def make_two_sip_model(self, use_xi_bar=False):
        """
        Makes the gurobi Model and returns it, given the data for a capacitated facility location problem
        as a CapFacLoc object.
        Data is an object of class CapFacLoc
        """
        n = self.num_nodes
        Nscen = self.num_scenarios
        c_f, c_v = self.c_f, self.c_v
        c_tf, c_tv = self.c_tf, self.c_tv

        if not use_xi_bar:
            d = self.Xi
            prob = np.array([1 / self.num_scenarios] * self.num_scenarios)
        else:
            d = self.xi_bar
            prob = np.asarray([1])
        MaxCap = np.max(np.sum(d, axis=1))

        model = Model()
        # Adding variables
        # Is there a facility at node i?
        b = model.addVars(n, obj=c_f, vtype=GRB.BINARY, name="b")
        # Capacity of facility at node i.
        v = model.addVars(n, obj=c_v, vtype=GRB.CONTINUOUS, name="v")
        y = tupledict()
        u = tupledict()
        xi_range = 1 if use_xi_bar else Nscen
        for i in range(n + 1):
            for j in range(n):
                for xi in range(xi_range):
                    # Is something sent from i to j in scenario xi?
                    y[i, j, xi] = model.addVar(vtype=GRB.CONTINUOUS, obj=c_tv[i, j] * prob[xi], name="y_" + str(i) + "_" + str(j) + "_" + str(xi))
                    # How much is sent from i to j in scenario xi.
                    u[i, j, xi] = model.addVar(vtype=GRB.BINARY, obj=c_tf[i, j] * prob[xi], name="u_" + str(i) + "_" + str(j) + "_" + str(xi))
        # Constraints
        # Cannot open too many facilities. not in more than 75% of the nodes
        model.addConstr(quicksum(b) <= 0.75 * n)
        # Cannot open too few facilities. Not in less than 10% of the nodes
        model.addConstr(quicksum(b) >= 0.1 * n)
        # No facility can have capacity more than MaxCap, and if b[i] is zero, there can't be a facility!
        model.addConstrs((v[i] <= MaxCap * b[i] for i in range(n)))

        # Scenario wise constraints
        # Total amount served from a facility is less than its capacity
        model.addConstrs((y.sum(i, '*', xi) <= v[i - 1] for i in range(1, n + 1) for xi in range(xi_range)))
        # Demand in each node is to be fulfilled
        model.addConstrs((y.sum('*', j, xi) == d[xi, j] for j in range(n) for xi in range(xi_range)))
        # If fixed cost is not paid (u[i,j,xi] is 0), transport cannot happen between i and j
        model.addConstrs((y[i, j, xi] <= u[i, j, xi] * MaxCap for i in range(n + 1) for j in range(n) for xi in range(
            xi_range)))
        model.addConstrs((u[i + 1, j, xi] <= b[i] for xi in range(xi_range) for j in range(n) for i in range(n)))

        return model, b, v, y, u

    def make_second_stage_model(self, sol, xi):
        model = Model()
        y = tupledict()
        u = tupledict()
        n = self.num_nodes
        c_tv, c_tf = self.c_tv, self.c_tf
        for i in range(n + 1):
            for j in range(n):
                y[i, j] = model.addVar(obj=c_tv[i, j], vtype=GRB.CONTINUOUS, name="y_" + str(i) + "_" + str(j))
                u[i, j] = model.addVar(obj=c_tf[i, j], vtype=GRB.BINARY, name="u_" + str(i) + "_" + str(j))
        model.addConstrs((y.sum(i, '*') <= sol['v'][i - 1] for i in range(1, n + 1)))
        model.addConstrs((y.sum('*', j) == self.Xi[xi, j] for j in range(n)))
        model.addConstrs((y[i, j] <= u[i, j] * self.MaxCap for i in range(n + 1) for j in range(n)))
        model.addConstrs((u[i + 1, j] <= sol['b'][i] for j in range(n) for i in range(n)))

        return model

    def solve_two_sip(self, use_xi_bar=False, gap=0.02, time_limit=600, threads=1):
        obj_time = []

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_time.append((model.cbGet(GRB.Callback.MIPSOL_OBJBST),
                                 model.cbGet(GRB.Callback.RUNTIME)))
                # print("Sol ", (model.cbGet(GRB.Callback.MIPSOL_OBJBST), "time ", model.cbGet(GRB.Callback.RUNTIME)))

        model, b, v, y, u = self.make_two_sip_model(use_xi_bar=use_xi_bar)
        model.setParam("LogToConsole", 0)
        model.setParam('MIPGap', gap)
        model.setParam('TimeLimit', time_limit)
        model.setParam('Threads', threads)
        model.update()
        model.optimize(callback)
        # print("1.", model.runTime)
        result = self.extract_extensive_result(model, b, v)
        result['obj_time'] = obj_time

        return result

    def get_second_stage_objective(self, sol, xi, threads=2):
        """
        Solves the small MIP for single second stage evaluation
        """
        model = self.make_second_stage_model(sol, xi)
        model.setParam("LogToConsole", 0)
        model.setParam("DualReductions", 0)
        model.setParam('Threads', threads)
        model.update()
        model.optimize()

        if model.getAttr('status') != 2:
            return GRB.INFINITY
        else:
            return model.getObjective().getValue()

    def evaluate_x(self, sol):
        """
        Evaluate the first stage decision (x)

        val = CF_objval(cflp_instance, sol)

        Given a first stage solution sol to a capacitated facility location problem cflp_instance, evaluates
        the objective value.

        Feasibility is not checked for sol (so that the function is fast!)
        """
        obj = np.inner(sol['b'], self.c_f) + np.inner(sol['v'], self.c_v)
        for xi in range(self.num_scenarios):
            second_stage_obj = self.get_second_stage_objective(sol, xi, threads=2)
            obj = obj + self.prob[xi] * second_stage_obj
        return obj

    @staticmethod
    def extract_extensive_result(model, b, v):
        """
        Extracts the stage 1 solution from a model and outputs it in a CF_Mastersol object
        M is Gurobi Model
        b and v are GRBVars*
        """
        val_v = model.getAttr('x', v)
        val_b = model.getAttr('x', b)
        vv = np.array([val_v[i] for i in val_v])
        bb = np.array([val_b[i] for i in val_b])
        # print("2.", model.runTime)
        return {'b': bb, 'v': vv, 'obj_val': model.objVal, 'obj_bound': model.objBound, 'gap': model.MIPGap}

    @staticmethod
    def initialize_result_xi(heuristic, xi_hat):
        return {
            'solved_surr': False,
            'solved_xi': False,
            'xi_hat': xi_hat,
            'exit_iteration': 0,
            'obj_val': -1,
            'mask_demand': heuristic.getint('mask_demand'),
            'average_reset': heuristic.getint('average_reset'),
            'average_reset_fraction': heuristic.getfloat('average_reset_fraction'),
            'linear_decrease': heuristic.getint('linear_decrease'),
            'linear_decrease_from': heuristic.getint('linear_decrease_from'),
            'linear_decrease_to': heuristic.getint('linear_decrease_to'),
            'custom_decrease': heuristic.getint('custom_decrease'),
            'is_improved': False
        }


def get_problem_identifier(problem_config):
    n_client = problem_config.getint('Problem', 'n_client')
    n_facility = problem_config.getint('Problem', 'n_facility')
    n_scenario = problem_config.getint('Problem', 'n_scenario')
    identifier = "_".join([str(n_client), str(n_facility), str(n_scenario)])

    return identifier

