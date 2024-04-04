

import logging
import math

import numpy as np
import pandas as pd

from datamemory import DataMemory, DataMemoryPointer
from nicdam.estimator import Estimator


TOLERANCE = 0.001

logger = logging.getLogger(__name__)


class NICDAM():
    def __init__(
            self,
            data_batch_size: int,
            stable_member: 'EnsembleMember',
            min_weight_stable_member: float,
            max_trained_on_data_rows: int,
            maturity_threshold_reactive_members: int=1000,
            max_n_reactive_members: int=3,
            accretion_rate_weights_reactive_members: float=0.01) -> None:
        """
        TODO: improve parameter naming?

        See uplift_ensemble_doc.md for details.

        Args:
            data_batch_size (int): The stream's expected data batch size.
            stable_member (EnsembleMember): The stable member of the ensemble.
            min_weight_stable_member (float): The constant minimum weight of the stable member.
            max_trained_on_data_rows (int): The maximum number of data rows a reactive member is
                trained on (more rows are not used for training).
            maturity_threshold_reactive_members (int): The number of data rows a reactive member
                needs to be trained on before it is considered mature. Defaults to 1000.
            max_n_reactive_members (int, optional): The maximum number of reactive members.
                Defaults to 3.
            accretion_rate_weights_reactive_members (float, optional): The accretion rate of
                reactive member weights. Defaults to 0.01.
        """
        self._data_memory = DataMemory(batch_size=data_batch_size)
        # TODO: accept a pre-existing DataMemory as input
        self._current_data_addendum = None

        # stable member
        self._stable_member = stable_member
        self._stable_member._ensemble_ref = self
        self._stable_member.set_data_memory_pointer(DataMemoryPointer(memory_ref=self._data_memory))
        # TODO: cases, where a DataMemory is passed as input and already has data in it should be
        #   used in full for stable member
        self._min_weight_stable_member = min_weight_stable_member

        # reactive members
        self._max_trained_on_data_rows = max_trained_on_data_rows
        # _maturity_threshold_reactive_members to separate max number of data points for training
        #   from the maturity definition
        self._maturity_threshold_reactive_members = maturity_threshold_reactive_members
        self._max_weight_sum_reactive_members = (1.0 - self._min_weight_stable_member)
        self._max_n_reactive_members = max_n_reactive_members
        self._accretion_rate_weights_reactive_members = accretion_rate_weights_reactive_members
        self._reactive_members = []

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if math.isclose(self._min_weight_stable_member, 1.0, abs_tol=TOLERANCE) \
                or self._min_weight_stable_member > 1.0:
            raise ValueError('Minimum weight of stable member must be less than 1.0.')
        if self._max_n_reactive_members < 1:
            raise ValueError('Maximum number of reactive members must be greater than 0.')
        if (self._max_n_reactive_members * self._accretion_rate_weights_reactive_members) > \
                (1.0 - self._min_weight_stable_member):
            raise ValueError('Accretion rate of reactive member weights multiplied by the maximum '
                             'number of reactive members must be less than the complement of the '
                             'minimum weight of the stable member.')

        if self._maturity_threshold_reactive_members > self._max_trained_on_data_rows:
            # otherwise, the maturity threshold is never reached
            self._maturity_threshold_reactive_members = self._max_trained_on_data_rows
            logger.warning('maturity_threshold_reactive_members cannot be greater than '
                           'max_trained_on_data_rows. It was set to the latter.')

    def fit_stable_member(self, data: pd.DataFrame, labels: pd.Series) -> None:
        # NOTE: the number of data rows the stable member is initially trained on is not limited
        addendum = pd.concat([data, labels], axis=1, sort=False, ignore_index=False)
        self._data_memory.append(new_data=addendum)
        self._stable_member.fit(data=data, labels=labels)

    def fit_incremental(
            self,
            addendum_data: pd.DataFrame,
            addendum_labels: pd.Series) -> None:
        # NOTE: when called, the internal data memory is augmented with the addendum
        addendum = pd.concat([addendum_data, addendum_labels], axis=1, sort=False, ignore_index=False)
        self._data_memory.append(new_data=addendum)
        self._current_data_addendum = addendum

        # prequential evaluation: score all members on the new data before training [Gama.2014?]
        self._score_members(data=addendum_data, labels=addendum_labels)

        for member in self.members:
            # NOTE: member.trained_on_n_data_rows can be greater than self._max_trained_on_data_rows
            #   due to the data_batch_size
            if member.trained_on_n_data_rows < self._max_trained_on_data_rows:
                member.fit_incremental()
            else:
                pass  # NOTE: member has no further training

        self._recalulate_weights_experience_based()

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        self._validate_weights()
        predictions = []
        for member in self.members:
            pred = member.predict(data=data)
            weighted_pred = pred * member.weight
            predictions.append(weighted_pred)
        if len(predictions) != self.n_members:
            raise RuntimeError(f'Unexpected number of predictions {len(predictions)}.')

        if len(predictions) == 1:
            ret = predictions[0]
        elif len(predictions) >= 1:
            ret = np.sum(predictions, axis=0)
        return ret

    def _score_members(self, data: pd.DataFrame, labels: pd.Series) -> None:
        for member in self.members:
            if member.trained_on_n_data_rows > 0:
                member.score(data=data, true_labels=labels)
            else:
                pass  # NOTE: reactive member has not begun training yet (just added)

    def add_reactive_member(
            self,
            estimator: Estimator,
            instantiation_t: int,
            rotate_members: str) -> None:
        if len(self._reactive_members) >= self._max_n_reactive_members:
            if rotate_members == 'oldest':
                # oldest reactive member is dropped
                self._drop_reactive_member(method='oldest')
            elif rotate_members == 'challenge stable or drop oldest':
                victorious_reactive_member = self._challenge_stable_member()
                if victorious_reactive_member is None:
                    # no mature reactive members present or no defeat of stable: drop oldest reactive member
                    self._drop_reactive_member(method='oldest')
                else:
                    # stable member was defeated by mature reactive member: replace stable member
                    self._replace_stable_by_reactive_member(new_stable_member=victorious_reactive_member)
            else:
                raise ValueError(f'Unknown value "{rotate_members}" for argument "rotate_members".')

        new_reactive_member = EnsembleMember(
            estimator=estimator,
            cols_data=self._stable_member._cols_data,
            col_label=self._stable_member._col_label,
            init_weight=0.0,
            instantiation_t=instantiation_t,
            ensemble_ref=self)
        new_reactive_member.set_data_memory_pointer(DataMemoryPointer(
            memory_ref=self._data_memory))
        # NOTE: The new DataMemoryPointer points to the NEXT batch of data that will be added to self._data_memory.
        # IMPORTANT: This assumes that the new DataMemoryPointer is created BEFORE the change-triggering batch of data is added to self._data_memory,
        #   if a change is the reason for calling this method.
        logger.info('Adding reactive member at instantiation_t = '
                    f'{new_reactive_member.instantiation_t}.')
        self._reactive_members.append(new_reactive_member)

        self._validate_weights()

    def _challenge_stable_member(self) -> 'EnsembleMember':
        highest_scoring_mature_reactive_member = self._get_highest_scoring_mature_reactive_member()
        if highest_scoring_mature_reactive_member is None:
            logger.info('No mature reactive members to challenge stable member present.')
            return None
        else:
            if highest_scoring_mature_reactive_member._get_aggregated_score(method='all') > \
                    self._stable_member._get_aggregated_score(method='all'):
                logger.info('Stable member was defeated by mature reactive member with '
                            f'instantiation_t = {highest_scoring_mature_reactive_member.instantiation_t}')
                return highest_scoring_mature_reactive_member
            else:
                logger.info('Stable member was not defeated by mature reactive member.')
                return None

    def _replace_stable_by_reactive_member(self, new_stable_member: 'EnsembleMember') -> None:
        self._reactive_members.remove(new_stable_member)
        new_weight = self._stable_member.weight + new_stable_member.weight
        # the sum of all members' weights remains unchanged --> no normalization needed

        self._stable_member.discontinue()
        self._stable_member = new_stable_member
        self._stable_member.weight = new_weight

    def _drop_reactive_member(self, method: str) -> None:
        if method == 'oldest':
            member_2_remove = self.get_oldest_reactive_member()
            if member_2_remove is None:
                raise RuntimeError('No reactive members present.')
        else:
            raise ValueError(f'Unknown value "{method}" for argument "method".')

        logger.info('Dropping reactive member with instantiation_t = '
                    f'{member_2_remove.instantiation_t}.')
        self._reactive_members.remove(member_2_remove)
        member_2_remove.discontinue()

        # NOTE: rationale: remaining reactive members get a share of the weight of the dropped member
        self._normalize_weights(members='reactive')

    def _normalize_weights(self, members='all') -> None:
        if members == 'all':
            members = self.members
            sum_weights_target = 1.0
        elif members == 'reactive':  # redistribute the complement of the stable member's weight
            members = self._reactive_members
            sum_weights_target = 1.0 - self._stable_member.weight
        else:
            raise ValueError(f'Unknown value "{members}" for argument "members".')

        sum_weights_actual = sum([member.weight for member in members])
        for member in members:
            member.weight = (member.weight / sum_weights_actual) * sum_weights_target

    def _recalulate_weights_experience_based(self) -> None:
        self._validate_weights()

        if not self._reactive_members:
            return

        sum_weights_reactive_members = sum([member.weight for member in self._reactive_members])
        if math.isclose(sum_weights_reactive_members, 0.0, abs_tol=TOLERANCE):  # initially, all reactive members have weight 0.0
            new_sum_weights_reactive_members = self._accretion_rate_weights_reactive_members
        else:
            new_sum_weights_reactive_members = min(
                1.0 - self._min_weight_stable_member,
                sum_weights_reactive_members * (1.0 + self._accretion_rate_weights_reactive_members))

        # update stable member's weight
        self._stable_member.weight = 1.0 - new_sum_weights_reactive_members

        if self.n_reactive_members == 1:
            self._reactive_members[0].weight = new_sum_weights_reactive_members
        else:  # > 1 reactive member
            # realize fast weight increase for most recent reactive member
            youngest_member = self.get_youngest_reactive_member()
            if math.isclose(youngest_member.weight, 0.0, abs_tol=TOLERANCE):  # initially
                youngest_member.weight = self._accretion_rate_weights_reactive_members
            else:
                youngest_member.weight = min(
                    youngest_member.weight * (1.0 + self._accretion_rate_weights_reactive_members),
                    self._max_weight_sum_reactive_members / self.n_reactive_members)  # not exceeding uniform weight distribution
            new_sum_weights_reactive_members = new_sum_weights_reactive_members - youngest_member.weight

            # distribute remaining weight among remaining reactive members
            remaining_members = [member for member in self._reactive_members
                                 if member is not youngest_member]
            sum_data_trained_on_remaining_members = sum([member.trained_on_n_data_rows
                                                         for member in remaining_members])
            for member in remaining_members:
                member.weight = (member.trained_on_n_data_rows / sum_data_trained_on_remaining_members) \
                    * new_sum_weights_reactive_members
            # TODO: some mechanics could be modified here to allow for steep weight increase for the youngest member:
            #   - member.trained_on_n_data_rows / sum_data_trained_on_remaining_members could be inverted by (1 - ...) in denominator
            #       --> steep increase (handle case where the weight of the youngest member is 0 initially)
            #   - possibly, the special treatment of the youngest member (youngest_member.weight = min(... ) could be removed

            try:
                self._validate_weights()
            except RuntimeError:
                logger.error(
                    'Weights needed to be normalized after experience-based weight update.')
                self._normalize_weights(members='reactive')

    def _validate_weights(self) -> None:
        for weight in [member.weight for member in self.members]:
            if not math.isclose(weight, 0.0, abs_tol=TOLERANCE) and weight < 0.0:
                raise RuntimeError('Member weight is negative.')
            if not math.isclose(weight, 1.0, abs_tol=TOLERANCE) and weight > 1.0:
                raise RuntimeError('Member weight is greater than 1.')

        weight_sum = sum([member.weight for member in self.members])
        if not math.isclose(weight_sum, 1.0, abs_tol=TOLERANCE):
            raise RuntimeError('Sum of member weights is not 1.')

        weight_sum_reactive_members = sum([member.weight for member in self._reactive_members])
        if not math.isclose(weight_sum_reactive_members, self._max_weight_sum_reactive_members,
                            abs_tol=TOLERANCE) \
                and weight_sum_reactive_members > self._max_weight_sum_reactive_members:
            raise RuntimeError('Sum of reactive member weights is greater than the maximum allowed.')

    def _get_mature_reactive_members(self) -> list['EnsembleMember']:
        return [member for member in self._reactive_members
                if member.trained_on_n_data_rows >= self._maturity_threshold_reactive_members]

    def _get_highest_scoring_mature_reactive_member(self) -> 'EnsembleMember':
        mature_members = self._get_mature_reactive_members()
        if not mature_members:
            return None
        else:
            highest_scoring_member = mature_members[0]
            for member in mature_members:
                if member._get_aggregated_score(method='all') > \
                        highest_scoring_member._get_aggregated_score(method='all'):
                    highest_scoring_member = member
            return highest_scoring_member

    @property
    def members(self) -> list['EnsembleMember']:
        return [self._stable_member] + self._reactive_members

    @property
    def reactive_members(self) -> list['EnsembleMember']:
        return self._reactive_members

    def get_youngest_reactive_member(self) -> 'EnsembleMember':
        if not self._reactive_members:
            return None
        else:
            youngest_member = self._reactive_members[0]
            for member in self._reactive_members:
                if member.instantiation_t > youngest_member.instantiation_t:
                    youngest_member = member
            return youngest_member

    def get_oldest_reactive_member(self) -> 'EnsembleMember':
        if not self._reactive_members:
            return None
        else:
            oldest_member = self._reactive_members[0]
            for member in self._reactive_members:
                if member.instantiation_t < oldest_member.instantiation_t:
                    oldest_member = member
            return oldest_member

    @property
    def n_members(self) -> int:
        return len(self.members)

    @property
    def n_reactive_members(self) -> int:
        return len(self._reactive_members)


class EnsembleMember:
    def __init__(
            self,
            estimator: Estimator,
            cols_data: list,
            col_label: str,
            init_weight: float,
            instantiation_t: int,
            ensemble_ref: NICDAM=None) -> None:
        self.estimator = estimator
        self._cols_data = cols_data
        self._col_label = col_label
        # estimator needs to implement the following methods:
        #   instantiate_estimator_model, preprocess_data(data), fit_model(data, labels),
        #   predict(data), score(true_labels, pred_labels) (maximizable)
        # estimator needs to implement the following attributes:
        #   estimator_type, n_data_rows_model_trained_on

        self.weight = init_weight
        self.instantiation_t = instantiation_t
        self._ensemble_ref = ensemble_ref

        self.trained_on_n_data_rows = 0

        self._scores = np.array([])

    def discontinue(self):
        """Clean-up steps when member is dropped from ensemble.
        """
        self.data_memory_pointer.unregister()

    def set_data_memory_pointer(self, data_memory_pointer) -> None:
        self.data_memory_pointer = data_memory_pointer

    def fit(self, data: pd.DataFrame, labels: pd.Series) -> None:
        data = data.loc[:, self._cols_data].copy(deep=True)

        self.estimator.instantiate_estimator_model()
        data = self.estimator.preprocess_data(data=data)
        self.estimator.fit_model(
            data=data,
            labels=labels)
        if len(data.index) != self.estimator.n_data_rows_model_trained_on:
            raise RuntimeError('Mismatch number of data rows trained on')
        else:
            self.trained_on_n_data_rows = self.estimator.n_data_rows_model_trained_on

    def fit_incremental(self) -> None:
        if self.estimator.estimator_type == 'batch':
            # based on [Gama.2014]: train from scratch
            data_memory_content = self.data_memory_pointer.get()  # all data from memory
            data = data_memory_content.loc[:, self._cols_data]
            labels = data_memory_content.loc[:, self._col_label]
            self.fit(data=data, labels=labels)
        elif self.estimator.estimator_type == 'incremental':
            if self._ensemble_ref._current_data_addendum is None:
                raise RuntimeError('No data addendum available.')
            data = self._ensemble_ref._current_data_addendum.loc[:, self._cols_data].copy(deep=True)
            labels = self._ensemble_ref._current_data_addendum.loc[:, self._col_label]
            data = self.estimator.preprocess_data(data=data)
            self.estimator.fit_model(
                data=data,
                labels=labels)
            if (len(data.index) + self.trained_on_n_data_rows) != self.estimator.n_data_rows_model_trained_on:
                raise RuntimeError('Mismatch number of data rows trained on')
            else:
                self.trained_on_n_data_rows = self.estimator.n_data_rows_model_trained_on
        else:
            raise RuntimeError(f'Unknown estimator type "{self.estimator.estimator_type}".')

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        data = self.estimator.preprocess_data(data=data.loc[:, self._cols_data].copy(deep=True))
        return self.estimator.predict(data=data)

    def score(
            self,
            data: pd.DataFrame,
            true_labels: pd.Series) -> None:
        pred_labels = self.predict(data=data)
        score = self.estimator.score(true_labels=true_labels, pred_labels=pred_labels)
        if self._scores.size == 0:
            self._scores = np.array([score])
        else:
            self._scores = np.append(self._scores, score)

    def _get_aggregated_score(
            self,
            method='all') -> float:
        if self._scores.size == 0:
            raise RuntimeError('No scores available.')

        if method == 'all':
            return np.mean(self._scores)
        else:
            raise ValueError(f'Unknown value "{method}" for argument "method".')


# custom exception for ensemble member instantiation
class EnsembleMemberInstantiationError(Exception):
    """Raised when an ensemble member cannot be instantiated."""
    pass
