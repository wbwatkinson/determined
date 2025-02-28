# coding: utf-8

"""
    Determined API (Beta)

    Determined helps deep learning teams train models more quickly, easily share GPU resources, and effectively collaborate. Determined allows deep learning engineers to focus on building and training models at scale, without needing to worry about DevOps or writing custom code for common tasks like fault tolerance or experiment tracking.  You can think of Determined as a platform that bridges the gap between tools like TensorFlow and PyTorch --- which work great for a single researcher with a single GPU --- to the challenges that arise when doing deep learning at scale, as teams, clusters, and data sets all increase in size.  # noqa: E501

    OpenAPI spec version: 0.1
    Contact: community@determined.ai
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


import pprint
import re  # noqa: F401

import six


class Trialv1Trial(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'id': 'int',
        'experiment_id': 'int',
        'start_time': 'datetime',
        'end_time': 'datetime',
        'state': 'Determinedexperimentv1State',
        'hparams': 'object',
        'total_batches_processed': 'int',
        'best_validation': 'V1MetricsWorkload',
        'latest_validation': 'V1MetricsWorkload',
        'best_checkpoint': 'V1CheckpointWorkload'
    }

    attribute_map = {
        'id': 'id',
        'experiment_id': 'experimentId',
        'start_time': 'startTime',
        'end_time': 'endTime',
        'state': 'state',
        'hparams': 'hparams',
        'total_batches_processed': 'totalBatchesProcessed',
        'best_validation': 'bestValidation',
        'latest_validation': 'latestValidation',
        'best_checkpoint': 'bestCheckpoint'
    }

    def __init__(self, id=None, experiment_id=None, start_time=None, end_time=None, state=None, hparams=None, total_batches_processed=None, best_validation=None, latest_validation=None, best_checkpoint=None):  # noqa: E501
        """Trialv1Trial - a model defined in Swagger"""  # noqa: E501

        self._id = None
        self._experiment_id = None
        self._start_time = None
        self._end_time = None
        self._state = None
        self._hparams = None
        self._total_batches_processed = None
        self._best_validation = None
        self._latest_validation = None
        self._best_checkpoint = None
        self.discriminator = None

        self.id = id
        self.experiment_id = experiment_id
        self.start_time = start_time
        if end_time is not None:
            self.end_time = end_time
        self.state = state
        self.hparams = hparams
        self.total_batches_processed = total_batches_processed
        if best_validation is not None:
            self.best_validation = best_validation
        if latest_validation is not None:
            self.latest_validation = latest_validation
        if best_checkpoint is not None:
            self.best_checkpoint = best_checkpoint

    @property
    def id(self):
        """Gets the id of this Trialv1Trial.  # noqa: E501

        The id of the trial.  # noqa: E501

        :return: The id of this Trialv1Trial.  # noqa: E501
        :rtype: int
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this Trialv1Trial.

        The id of the trial.  # noqa: E501

        :param id: The id of this Trialv1Trial.  # noqa: E501
        :type: int
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")  # noqa: E501

        self._id = id

    @property
    def experiment_id(self):
        """Gets the experiment_id of this Trialv1Trial.  # noqa: E501

        The id of the parent experiment.  # noqa: E501

        :return: The experiment_id of this Trialv1Trial.  # noqa: E501
        :rtype: int
        """
        return self._experiment_id

    @experiment_id.setter
    def experiment_id(self, experiment_id):
        """Sets the experiment_id of this Trialv1Trial.

        The id of the parent experiment.  # noqa: E501

        :param experiment_id: The experiment_id of this Trialv1Trial.  # noqa: E501
        :type: int
        """
        if experiment_id is None:
            raise ValueError("Invalid value for `experiment_id`, must not be `None`")  # noqa: E501

        self._experiment_id = experiment_id

    @property
    def start_time(self):
        """Gets the start_time of this Trialv1Trial.  # noqa: E501

        The time the trial was started.  # noqa: E501

        :return: The start_time of this Trialv1Trial.  # noqa: E501
        :rtype: datetime
        """
        return self._start_time

    @start_time.setter
    def start_time(self, start_time):
        """Sets the start_time of this Trialv1Trial.

        The time the trial was started.  # noqa: E501

        :param start_time: The start_time of this Trialv1Trial.  # noqa: E501
        :type: datetime
        """
        if start_time is None:
            raise ValueError("Invalid value for `start_time`, must not be `None`")  # noqa: E501

        self._start_time = start_time

    @property
    def end_time(self):
        """Gets the end_time of this Trialv1Trial.  # noqa: E501

        The time the trial ended if the trial is stopped.  # noqa: E501

        :return: The end_time of this Trialv1Trial.  # noqa: E501
        :rtype: datetime
        """
        return self._end_time

    @end_time.setter
    def end_time(self, end_time):
        """Sets the end_time of this Trialv1Trial.

        The time the trial ended if the trial is stopped.  # noqa: E501

        :param end_time: The end_time of this Trialv1Trial.  # noqa: E501
        :type: datetime
        """

        self._end_time = end_time

    @property
    def state(self):
        """Gets the state of this Trialv1Trial.  # noqa: E501

        The current state of the trial.  # noqa: E501

        :return: The state of this Trialv1Trial.  # noqa: E501
        :rtype: Determinedexperimentv1State
        """
        return self._state

    @state.setter
    def state(self, state):
        """Sets the state of this Trialv1Trial.

        The current state of the trial.  # noqa: E501

        :param state: The state of this Trialv1Trial.  # noqa: E501
        :type: Determinedexperimentv1State
        """
        if state is None:
            raise ValueError("Invalid value for `state`, must not be `None`")  # noqa: E501

        self._state = state

    @property
    def hparams(self):
        """Gets the hparams of this Trialv1Trial.  # noqa: E501

        Trial hyperparameters.  # noqa: E501

        :return: The hparams of this Trialv1Trial.  # noqa: E501
        :rtype: object
        """
        return self._hparams

    @hparams.setter
    def hparams(self, hparams):
        """Sets the hparams of this Trialv1Trial.

        Trial hyperparameters.  # noqa: E501

        :param hparams: The hparams of this Trialv1Trial.  # noqa: E501
        :type: object
        """
        if hparams is None:
            raise ValueError("Invalid value for `hparams`, must not be `None`")  # noqa: E501

        self._hparams = hparams

    @property
    def total_batches_processed(self):
        """Gets the total_batches_processed of this Trialv1Trial.  # noqa: E501

        The current processed batches.  # noqa: E501

        :return: The total_batches_processed of this Trialv1Trial.  # noqa: E501
        :rtype: int
        """
        return self._total_batches_processed

    @total_batches_processed.setter
    def total_batches_processed(self, total_batches_processed):
        """Sets the total_batches_processed of this Trialv1Trial.

        The current processed batches.  # noqa: E501

        :param total_batches_processed: The total_batches_processed of this Trialv1Trial.  # noqa: E501
        :type: int
        """
        if total_batches_processed is None:
            raise ValueError("Invalid value for `total_batches_processed`, must not be `None`")  # noqa: E501

        self._total_batches_processed = total_batches_processed

    @property
    def best_validation(self):
        """Gets the best_validation of this Trialv1Trial.  # noqa: E501

        Best validation.  # noqa: E501

        :return: The best_validation of this Trialv1Trial.  # noqa: E501
        :rtype: V1MetricsWorkload
        """
        return self._best_validation

    @best_validation.setter
    def best_validation(self, best_validation):
        """Sets the best_validation of this Trialv1Trial.

        Best validation.  # noqa: E501

        :param best_validation: The best_validation of this Trialv1Trial.  # noqa: E501
        :type: V1MetricsWorkload
        """

        self._best_validation = best_validation

    @property
    def latest_validation(self):
        """Gets the latest_validation of this Trialv1Trial.  # noqa: E501

        Latest validation.  # noqa: E501

        :return: The latest_validation of this Trialv1Trial.  # noqa: E501
        :rtype: V1MetricsWorkload
        """
        return self._latest_validation

    @latest_validation.setter
    def latest_validation(self, latest_validation):
        """Sets the latest_validation of this Trialv1Trial.

        Latest validation.  # noqa: E501

        :param latest_validation: The latest_validation of this Trialv1Trial.  # noqa: E501
        :type: V1MetricsWorkload
        """

        self._latest_validation = latest_validation

    @property
    def best_checkpoint(self):
        """Gets the best_checkpoint of this Trialv1Trial.  # noqa: E501

        Best checkpoint.  # noqa: E501

        :return: The best_checkpoint of this Trialv1Trial.  # noqa: E501
        :rtype: V1CheckpointWorkload
        """
        return self._best_checkpoint

    @best_checkpoint.setter
    def best_checkpoint(self, best_checkpoint):
        """Sets the best_checkpoint of this Trialv1Trial.

        Best checkpoint.  # noqa: E501

        :param best_checkpoint: The best_checkpoint of this Trialv1Trial.  # noqa: E501
        :type: V1CheckpointWorkload
        """

        self._best_checkpoint = best_checkpoint

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(Trialv1Trial, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, Trialv1Trial):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
