# -*- coding: utf-8 -*-
"""
MLMC storage dispatch case study
@author: Simon Tindemans
Delft University of Technology
s.h.tindemans@tudelft.nl

Functions AI_n_store_generator() and run_greedy_AI() by Ensieh Sharifnia, TU Delft
Function optimal_n_store_generator() by Michael Evans, Imperial College London.


This code implements the case study in the paper
"Multilevel Monte Carlo with Surrogate Models for Resource Adequacy Assessment",
Ensieh Sharifnia and Simon Tindemans,
accepted for publication at PMAPS 2022.
A preprint is available at: arXiv:2203.03437

If you use (parts of) this code, please cite the preprint or published paper.
"""
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import quadprog

# import base class definitions for multi-level sampling
import gen_adequacy

import MLSampleBase
# import base class for machine learning algorithm
import MachineLearning


EFFECTIVE_ZERO = 1e-10

def define_storage(store_total_power, store_duration, store_units, store_dispersion):
    """
    Initialise contribution of storage units

    :param store_total_power: total power of storage units
    :param store_duration: mean duration (at max power) of storage units
    :param store_units: number of storage units
    :param store_dispersion: dispersion of duration
    :return: None
    """

    if store_units == 1:
        store_power_list = np.array([store_total_power])
        store_energy_list = np.array([store_duration * store_total_power])
    else:  # create diverse array of storage units
        store_power_list = np.full(shape=(store_units), fill_value=store_total_power / store_units)
        store_energy_list = np.linspace(
            (1 + store_dispersion) * store_total_power * store_duration / store_units,
            (1 - store_dispersion) * store_total_power * store_duration / store_units,
            num=store_units)

    return store_power_list, store_energy_list



class StorageSystem(MLSampleBase.MLSampleFactory):
    """
    Storage System class definition

    """

    def __init__(self, demand_samples, wind_samples,
                 store_power_list=None, store_energy_list=None, store_efficiency=1.0, train_size=500):
        """
        Initialise storage system

        :param demand_samples: dictionary of {year: time series} pairs
        :param wind_samples:  dictionary of {year: time series} pairs
        :param store_power_list: list of max power of storage units
        :param store_energy_list: list of max energy of storage units
        :param store_efficiency: efficiency of charging
        """

        if store_efficiency != 1.0:
            raise NotImplementedError()

        # Call superclass constructor, announcing available sample levels, accepted combinations and outputs.
        super(self.__class__, self).__init__(
            output_labels=('LOLE', 'EENS'),
            output_units=('h', 'MWh'),
            available_levels=('OptimalNStore', 'GreedyNStore', 'AIGreedyNStore', 'Greedy1Store', 'AvgStore', 'NoStore', 'AIModel'),
            suggested_hierarchy=('OptimalNStore', 'AIGreedyNStore', 'AIModel', 'AvgStore'),
            permissible_level_sets=[
                {'OptimalNStore', 'GreedyNStore', 'AIGreedyNStore', 'Greedy1Store', 'AvgStore', 'NoStore', 'AIModel'},
                ]
        )

        self.demand_samples = demand_samples
        self.wind_samples = wind_samples

        # Generate net demand traces from all permutations of demand traces and wind power traces
        net_demand_sample_combinations = [demand_samples[demand_year] - wind_samples[wind_year]
                                          for demand_year in demand_samples
                                          for wind_year in wind_samples]
        # initialise a system that has 3 hours LOLE across all net demand sample traces
        self.rv_system = gen_adequacy.autogen_system(load_profile=np.concatenate(net_demand_sample_combinations),
                                                      wind_profile=None,
                                                      LOLH=3,
                                                      gen_set=[1200, 600, 600, 250, 250, 120, 60, 20, 20, 10, 10, 10],
                                                      MTBF=2000,
                                                      apply_load_offset=True,
                                                      resolution=10, gen_availability=0.90)

        # manually apply load offset to gross and net demand samples to hit 3 hours LOLE in the reference scenario
        self.demand_samples = {year: demand_samples[year] + self.rv_system.load_offset for year in self.demand_samples}

        # initialise storage units
        assert (store_power_list is None) == (store_energy_list is None), \
            "store_power_list and store_energy_list must both be specified or not at all."
        if store_power_list is not None:
            assert len(store_power_list) == len(store_energy_list), \
                "store_power_list and store_energy_list must have equal length"

        if store_power_list is None:
            self.store_power_list = np.array([0.0])
            self.store_energy_list = np.array([0.0])
        else:
            self.store_power_list = np.array(store_power_list)
            self.store_energy_list = np.array(store_energy_list)

        # compute the average storage response of a single unit
        mean_daily_demand = np.mean(
            [self.demand_samples[year].reshape((-1, 24)).mean(axis=0) for year in self.demand_samples],
            axis=0)

        mean_daily_demand_with_storage = self._load_flattener_periodic(mean_daily_demand,
                                                                 power_limit=np.sum(self.store_power_list),
                                                                 energy_limit=np.sum(self.store_energy_list))
        daily_storage_demand_offset = mean_daily_demand_with_storage - mean_daily_demand
        self.avg_storage_demand_offset_trace = np.tile(daily_storage_demand_offset, 365)
        # AI model
        self.AI_model = MachineLearning.MachineLearning(train_size= train_size)

        return


    def generate_sample(self, level_set):
        """
        Implement base class function to generate sample objects
        :param level_set: index set to support
        :return: StorageSample object (derived from MLSample)
        """
        return StorageSample(level_set=level_set, power_system=self)


    def expectation_value(self, level):
        """
        Implement base class function to compute expectation values directly (without sampling)
        :param level: single index
        :return: expectation value
        """
        if level == 'OptimalNStore':
            raise NotImplementedError
        if level == 'GreedyNStore':
            raise NotImplementedError
        elif level == 'AIGreedyNStore':
            raise NotImplementedError
        elif level == 'Greedy1Store':
            raise NotImplementedError
        elif level == 'AvgStore':
            return self._expectation_avg()
        elif level == 'NoStore':
            return self._expectation_no_store()
        elif level == 'AIModel':
            raise NotImplementedError
        else:
            raise NotImplementedError


    def expectation_value_available(self, level):
        """
        Implement base class function that specifies whether analytical expectation values are available
        :param level: single index
        :return: boolean
        """
        if level == 'OptimalNStore':
            return False
        elif level == 'GreedyNStore':
            return False
        elif level == 'AIGreedyNStore':
            return False
        elif level == 'Greedy1Store':
            return False
        elif level == 'AvgStore':
            return True
        elif level == 'NoStore':
            return True
        elif level == 'AIModel':
            return False
        else:
            raise NotImplementedError


    def _expectation_no_store(self, _cache={}):
        """
        Internal function to return expectation values for the NoStore reference
        :param _cache:
        :return: expectation value pair (LOLP, EENS)
        """
        # include dirty memoization hack to store previously computed result
        if len(_cache) == 0:
            _cache[0] = np.array((8760 * self.rv_system.lolp(), 8760 * self.rv_system.epns()))
        return _cache[0]


    def _expectation_avg(self, _cache={}):
        """
        Internal function to return expectation values for the AvgStore case
        :param _cache:
        :return: expectation value pair (LOLP, EENS)
        """
        # include dirty memoization hack to store previously computed result
        if len(_cache) == 0:



            # generate all net demand traces with inclusion of identical daily storage dispatch
            adjusted_net_demand_samples = [self.demand_samples[demand_year] + self.avg_storage_demand_offset_trace - self.wind_samples[wind_year]
                                           for demand_year in self.demand_samples
                                           for wind_year in self.wind_samples]
            # create a gen_adequacy.SingleNodeSystem and use internal tools to compute LOLP and EENS by convolution
            adjusted_rv_system = gen_adequacy.SingleNodeSystem(gen_list=self.rv_system.gen_list,
                                                                  load_profile=np.concatenate(adjusted_net_demand_samples),
                                                                  wind_profile=None,
                                                                  resolution=self.rv_system.resolution,
                                                                  load_offset=None
                                                                  )
            _cache[0] = np.array((8760 * adjusted_rv_system.lolp(), 8760 * adjusted_rv_system.epns()))
        return _cache[0]


    def _load_flattener_periodic(self, load_vec, power_limit, energy_limit):
        """
        Function that determines a peak-shaving/valley-filling dispatch pattern for the battery, with periodic boundaries.

        :param load_vec: load profile
        :param power_limit: max power
        :param energy_limit: max energy
        :return: flattened load profile (original + battery)

        I selected this optimiser after reading:
        https://scaron.info/blog/quadratic-programming-in-python.html

        The variable vector consists of 1 initial (and final) energy level and N power levels (for each time stamp)
        """
        batch_size = len(load_vec)

        G_mat = np.identity(batch_size + 1)
        G_mat[0, 0] = 0.000001  # NOTE: this should be zero, but the solver requires positive definite G

        if isinstance(load_vec, pd.Series):
            a_vec = -load_vec.values
        else:
            a_vec = -load_vec
        a_vec = np.insert(a_vec, 0, 0)

        equality_constraint = np.ones((1, batch_size + 1))
        equality_constraint[0, 0] = 0
        power_constraints = np.vstack((np.eye(batch_size, batch_size + 1, 1), -np.eye(batch_size, batch_size + 1, 1)))
        energy_constraints = np.vstack((np.tri(batch_size, batch_size + 1), -np.tri(batch_size, batch_size + 1)))

        C_mat = np.vstack((equality_constraint, power_constraints, energy_constraints)).transpose()
        b_vec = -np.concatenate(
            (np.zeros(1), np.ones(2 * batch_size) * power_limit, np.zeros(batch_size), np.ones(batch_size) * energy_limit))

        power_solution = quadprog.solve_qp(G=G_mat, a=a_vec, C=C_mat, b=b_vec, meq=1, factorized=False)[0][1:]

        return load_vec + power_solution


    def generate_margin_trace(self):
        # select a random demand year and its corresponding demand trace
        demand_trace = self.demand_samples[np.random.choice(list(self.demand_samples))]
        # select a random wind trace
        wind_trace = self.wind_samples[np.random.choice(list(self.wind_samples))]
        # generate a random thermal generation trace
        thermal_gen_trace = self.rv_system.generation_trace(num_steps=8760)
        # compute the resulting margin trace
        return thermal_gen_trace + wind_trace - demand_trace


    def generate_daily_margin_trace(self):
        # select a random demand year
        demand_year = np.random.choice(list(self.demand_samples))
        # select a random wind year
        wind_year = np.random.choice(list(self.wind_samples))
        # select a random date
        date = np.random.randint(0, 364)
        # select the corresponding demand trace
        demand_trace = self.demand_samples[demand_year][date * 24:(date + 1) * 24]
        # select the corresponding wind trace
        wind_trace = self.wind_samples[wind_year][date * 24:(date + 1) * 24]
        # generate a random thermal generation trace
        thermal_gen_trace = self.rv_system.generation_trace(num_steps=24)
        # compute the resulting margin trace
        return thermal_gen_trace + wind_trace - demand_trace


    def greedy_storage_dispatch(self, pre_margin, power_limit, energy_limit, dt=1):
        """
        Execute greedy storage algorithm on a net margin time trace

        :param pre_margin: margin time series without storage
        :param power_limit: max power (charge and discharge assumed identical)
        :param energy_limit: max energy storage
        :param dt: time step (units of energy/power)
        :return: adjusted margin time series
        """
        assert power_limit > 0
        assert energy_limit > 0

        # assume start with a full store
        energy_stored = energy_limit

        post_margin = np.zeros(pre_margin.size)
        for t, margin in enumerate(pre_margin):
            store_power = max(min(margin, power_limit, (energy_limit - energy_stored) / dt), -power_limit,
                              -energy_stored / dt)
            energy_stored += store_power
            post_margin[t] = margin - store_power

        return post_margin
    

    def run_greedy_policy(self, margin_trace):
        '''
        Compute (LOL, ENS) for sample in the Gre model
        Parameters: 
            margin_trace: array
        Return:
            (LOL, ENS) array           
        '''
        # generate a list of units and sort them by time to go
        store_list = [item for item in zip(self.store_power_list, self.store_energy_list)]
        store_list.sort(key=lambda x: x[1]/x[0], reverse=True)
        for unit_power, unit_energy in store_list:
            margin_trace = self.greedy_storage_dispatch(pre_margin=margin_trace,
                                                                    power_limit=unit_power,
                                                                    energy_limit=unit_energy)
       
        shortfalls = margin_trace < 0
        lol = np.sum(shortfalls)
        ue = -np.sum(margin_trace[shortfalls])
        return np.array([lol, ue])


    def run_greedy_AI(self, margin_trace):
        '''
        Compute (LOL, ENS) for sample in the HGB+Gre model
        Parameters: 
            margin_trace: array
        Return:
            (LOL, ENS) array
        Implemented by Ensieh Sharifnia, TU Delft           
        '''
        margin_trace = np.reshape(margin_trace, (-1,24))
        lol_estimate = self.AI_model.predict(margin_trace, target=1)[0].astype(int)
        margin_trace = margin_trace[lol_estimate>EFFECTIVE_ZERO]
        margin_trace= margin_trace.ravel()
        return self.run_greedy_policy(margin_trace)


    def run_optimal_policy(self, margin_trace):
        """
        Compute (LOL, ENS) for sample in the Exact model
        :return: (LOL, ENS) array

        Implemented by Michael Evans, Imperial College London. Refactored by Simon Tindemans, TU Delft.
        """

        def _optimal_discharge_policy_step(shortfall, state, power_limit, dt):
            """
            Greedy optimal dispatch policy for n devices; for a single time step.

            :param shortfall: current shortfall without storage
            :param initial_state: duration state vector across stores
            :param power_limit: max power vector across stores
            :param dt: time step (units of energy/power)
            :return: optimal dispatch
            """
            n = len(state)
            one = np.ones([n, 1])  # unity vector (length n)
            zero = np.zeros([n, 1])  # zero vector (length n)
            y = np.unique(np.sort(np.concatenate([state, np.max([state - dt * one, zero], axis=0)], axis=0)))
            y = y[::-1]  # decreasing order
            E_u = 0
            i = -1  # to allow for 0-indexing
            while True:
                i += 1
                E_l = E_u
                E_u = np.sum(
                    np.multiply(power_limit,
                                np.max([np.min([state - y[i] * one, dt * one], axis=0), zero], axis=0)))
                if E_u >= shortfall * dt or i == len(y) - 1:
                    break
            if E_u <= shortfall * dt:
                z_hat = y[i]
            else:
                z_hat = y[i - 1] + (shortfall * dt - E_l) * (y[i] - y[i - 1]) / (E_u - E_l)
            u = np.multiply(power_limit, np.max([np.min([(state - z_hat * one) / dt, one], axis=0), zero], axis=0))
            return u

        def _recharge_policy_step(excess, state, duration_limit, power_limit_up, power_limit_down, eta, dt):
            """
            Greedy recharge policy for n devices; for a single time step.

            :param excess: current excess without storage
            :param initial_state: duration state vector across stores
            :param duration_limit: maximum duration limit across stores
            :param power_limit: max power vector across stores
            :param dt: time step (units of energy/power)
            :return: recharge dispatch
            """
            n = len(state)
            one = np.ones([n, 1])  # unity vector (length n)
            zero = np.zeros([n, 1])  # zero vector (length n)
            z_bar = np.min([state - eta * power_limit_up * dt / power_limit_down, duration_limit], axis=0)
            y = np.unique(np.sort(np.concatenate([state, z_bar], axis=0), axis=0))  # ascending order
            E_u = 0
            i = -1  # to allow for 0-indexing
            while True:
                i += 1
                E_l = E_u
                E_u = np.sum(np.multiply(power_limit_down,
                                         np.max(
                                             [np.min([y[i] * one - state, z_bar - state, dt * one], axis=0), zero],
                                             axis=0)))
                if E_u >= excess * dt or i == len(y) - 1:
                    break
            if E_u <= excess * dt:
                z_hat = y[i]
            else:
                z_hat = y[i - 1] + (excess * dt - E_l) * (y[i] - y[i - 1]) / (E_u - E_l)
            u = np.multiply(power_limit_down,
                            np.min([np.max([(state - z_hat * one) / dt, (state - z_bar) / dt], axis=0), zero],
                                   axis=0)) / eta
            return u

        lol = 0
        eu = 0.0
        energy_limits = np.array([self.store_energy_list]).transpose()
        power_limits = np.array([self.store_power_list]).transpose()
        state = energy_limits / power_limits  # assume batteries start full
        for ts in range(0, len(margin_trace)):
            if margin_trace[ts] <= 0:
                control_input = _optimal_discharge_policy_step(-margin_trace[ts], state, power_limits, dt=1)
                ens = -margin_trace[ts] - np.sum(control_input)
                lol += (ens > EFFECTIVE_ZERO)
                eu += ens
            else:
                control_input = _recharge_policy_step(margin_trace[ts], state, energy_limits / power_limits,
                                                      -power_limits, power_limits,
                                                      eta=1, dt=1)
            state = state - control_input / power_limits
        return np.array([lol, eu])
    

class StorageSample(MLSampleBase.MLSample):
    """
    Class for a single sample (i.e. realisation) of annual loss of load.

    This derives from MSSampleBase.MLSample for compatibility with the MLMC framework.
    """
    def __init__(self, level_set, power_system):
        """
        Initialise sample

        :param level_set: index set to support
        :param power_system: reference to the power system object
        """
        # Required function: formally initialise sample by instantiating random realisation.
        super(self.__class__, self).__init__(level_set=level_set)
        self.system = power_system
        self.margin_trace = self.system.generate_margin_trace()

        return


    def generate_value(self, level):
        """
        Generate sample outputs at specified index level. [Required implementation for MLSample base class]

        :param level: specific index
        :return: (LOL, ENS) array
        """

        if level == 'OptimalNStore':
            return self.optimal_n_store_generator()
        elif level == 'GreedyNStore':
            return self.greedy_n_store_generator()
        elif level == 'AIGreedyNStore':
            return self.greedy_AI_generator()
        elif level == 'Greedy1Store':
            return self.greedy_1_store_generator()
        elif level == 'AvgStore':
            return self.avg_store_generator()
        elif level == 'NoStore':
            return self.no_store_generator()
        elif level == 'AIModel':
            return self.AI_n_store_generator()
        else:
            raise RuntimeError()


    def no_store_generator(self):
        """
        Compute (LOL, ENS) for sample for the NoStore model

        :return: (LOL, ENS) array
        """
        shortfalls = self.margin_trace < 0
        lol = np.sum(shortfalls)
        ue = -np.sum(self.margin_trace[shortfalls])
        return np.array([lol, ue])


    def avg_store_generator(self):
        """
        Compute (LOL, ENS) for sample in the AvgStore model
        :return: (LOL, ENS) array
        """

        adjusted_margin_trace = self.margin_trace - self.system.avg_storage_demand_offset_trace

        shortfalls = adjusted_margin_trace < 0
        lol = np.sum(shortfalls)
        ue = -np.sum(adjusted_margin_trace[shortfalls])
        return np.array([lol, ue])


    def greedy_1_store_generator(self):
        """
        Compute (LOL, ENS) for sample in the Greedy1Store model
        :return: (LOL, ENS) array
        """
        # efficiency measure: evaluate AvgStore policy. If it results in no impacts, skip further evaluation
        if (self['AvgStore'] < EFFECTIVE_ZERO).all():
            return np.array([0., 0.])

        # apply greedy storage to margin trace
        post_margin_trace = self.system.greedy_storage_dispatch(pre_margin=self.margin_trace,
                                                                 power_limit=np.sum(self.system.store_power_list),
                                                                 energy_limit=np.sum(self.system.store_energy_list))

        shortfalls = post_margin_trace < 0
        lol = np.sum(shortfalls)
        ue = -np.sum(post_margin_trace[shortfalls])
        return np.array([lol, ue])


    def greedy_n_store_generator(self):
        """
        Compute (LOL, ENS) for sample in the GreedyNStore model
        :return: (LOL, ENS) array
        """
        if (self['NoStore'] < EFFECTIVE_ZERO).all():
            return np.array([0., 0.])
        return self.system.run_greedy_policy(self.margin_trace)
    

    def greedy_AI_generator(self):
        """
        Compute (LOL, ENS) for sample in the GreedyNStore model
        :return: (LOL, ENS) array
        """
        if (self['NoStore'] < EFFECTIVE_ZERO).all():
            return np.array([0., 0.])
        return self.system.run_greedy_AI(self.margin_trace)


    def optimal_n_store_generator(self):
        """
        Compute (LOL, ENS) for sample in the OptimalNStore model
        :return: (LOL, ENS) array

        Implemented by Michael Evans, Imperial College London. Refactored by Simon Tindemans, TU Delft.
        """

        return self.system.run_optimal_policy(self.margin_trace)


    def AI_n_store_generator(self):
        """
        Compute (LOL, ENS) for sample in the surrogate model
        :return: (LOL, ENS) array
        Implemented by Ensieh Sharifnia, TU Delft
        """
        daily_margin_traces = np.reshape(self.margin_trace, (-1,24))
        
        if (daily_margin_traces.size > 0):
            lol, ens = self.system.AI_model.predict(daily_margin_traces, target=2)
            return np.array([np.sum(lol), np.sum(ens)]) 
        else:
            return np.array([0., 0.])
        
         




if __name__ == "__main__":

    import MCCoordinator


    def storage_system(**kwargs):
        """
        Instantiate StorageSystem object using GB system data

        :param wind_power: assumed wind power capacity (in MW)
        :param kwargs: additional arguments to be supplied to the StorageSystem constructor
        :return: StorageSystem object
        """

        data = pd.read_csv('../data/UKdata/20161213_uk_wind_solar_demand_temperature.csv',
                           parse_dates=['UTC Time', 'Local Time'], infer_datetime_format=True, dayfirst=True,
                           index_col=0)

        demand_data = data['demand_net'].dropna()['2006':'2015']
        wind_data = 10000 * data['wind_merra1'].dropna()

        demand_samples = {yeardata[0]: yeardata[1].values[:8760] for yeardata in
                          demand_data.groupby(demand_data.index.year)}
        wind_samples = {yeardata[0]: yeardata[1].values[:8760] for yeardata in wind_data.groupby(wind_data.index.year)}

        dataframe = pd.read_csv('../data/battery_data.csv', index_col=0)
        store_power_list = 3 * dataframe['Power (MW)']
        store_energy_list = 3 * dataframe['Energy (MWh)']

        return StorageSystem(demand_samples=demand_samples,
                                            wind_samples=wind_samples,
                                            store_power_list=store_power_list,
                                            store_energy_list=store_energy_list,
                                            **kwargs)


    system = storage_system()

    # set up MLMC coordinator with link to system
    mcc = MCCoordinator.MCCoordinator(factory=system,
                                      ml_hierarchy=['OptimalNStore', 'GreedyNStore', 'AvgStore'],
                                      use_expectations=True,
                                      use_joblib=False, joblib_n_jobs=-1, joblib_batch_size=5)

    mcc.explore(n_samples=5)

    for i in range(10):
        mcc.run_recommended(time_seconds=10, verbose=True, optimization_target='EENS')

    mcc.verbose_result()
